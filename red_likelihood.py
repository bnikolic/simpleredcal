"""Robust redundant calibration"""


import os
import functools
from copy import deepcopy

import numpy
from matplotlib import pyplot as plt
from scipy.optimize import Bounds, minimize
from scipy.special import gamma
from scipy.stats import circmean

import hera_cal
from hera_cal.io import HERACal, HERAData
from hera_cal.noise import predict_noise_variance_from_autos
from hera_cal.redcal import get_reds

from jax.config import config
config.update('jax_enable_x64', True)
import jax
from jax import jit, jacrev, jacfwd
from jax import numpy as np
from jax.scipy.optimize import minimize as jminimize

# n.b. where 'numpy' is used below it has to be real numpy. 'np' here is the
# jax implementation


def fltBad(bll, badl, minbl=2):
    """Filter bad baselines

    :param bll: List of redundant baseline sets, as returned by get_reds
    :type: bll: list of list of tuples
    :param badl: List of bad antennas
    :type badl: list
    :param minbl: Minimum number of observed baselines in a group
    :type minbl: int

    :return: Filtered list of baselines
    :rtype: list
    """
    r1 = map(functools.partial(filter, lambda x: not (x[0] in badl or x[1] \
                               in badl)), bll)
    r2 = list(map(list, r1))
    return list(filter(lambda x: len(x) >= minbl, r2))


def groupBls(bll):
    """Map baselines to redundant baseline indices

    :param bll: List of redundant baseline sets, as returned by get_reds
    :type: bll: list of list of tuples

    :return: Array of dimensions (total_no_baselines, 3), where the format of
    each row is [group_id, ant_i, ant_j]
    :rtype: ndarray
    """
    return np.array([(g, i, j) for (g, bl) in enumerate(bll) for (i, j, p) in bl])


def condenseMap(arr):
    """Return a mapping of indices to a dense index set

    :param arr: Array
    :type arr: ndarray

    :return: Dictionary that maps unique elements to a dense index set
    :rtype: dict
    """
    return dict(map(reversed, enumerate(numpy.unique(arr))))


def relabelAnts(redg):
    """Relabel antennas with consecutive numbering

    :param redg: Grouped baselines, as returned by groupBls
    :type redg: ndarray

    :return: Relabelled grouped baselines
    :rtype: ndarray
    """
    ci = condenseMap(redg[:, 1:3])
    redg = numpy.copy(redg)
    for i in range(len(redg)):
        redg[i, 1] = ci[redg[i, 1]]
        redg[i, 2] = ci[redg[i, 2]]
    return redg


def group_data(zen_path, pol, chans=None, tints=None, bad_ants=None, \
               flag_path=None, noise=False):
    """Returns redundant baseline grouping and reformatted dataset, with
    external flags applied, if specified

    :param zen_path: Path of uvh5 dataset
    :type zen_path: str
    :param pol: Polarization of data
    :type pol: str
    :param chans: Frequency channel(s) {0, 1023} (None to choose all)
    :type chans: array-like, int, or None
    :param tints: Time integrations {0, 59} (None to choose all)
    :type tints: array-like, int, or None
    :param bad_ants: Known bad antennas to flag, optional
    :type bad_ants: array-like, None
    :param flag_path: Path of calfits flag file, optional
    :type flag_path: str, None
    :param noise: Also calculate noise from autocorrelations
    :type nois: bool

    :return hd: HERAData class
    :rtype hd: HERAData class
    :return redg: Grouped baselines, as returned by groupBls
    :rtype redg: ndarray
    :return cdata: Grouped visibilities with flags in numpy MaskedArray format,
    with format consistent with redg and dimensions (freq chans,
    time integrations, baselines)
    :rtype cdata: MaskedArray
    :return cndata: Grouped noise, with same dimensions as cdata
    :rtype cndata: ndarray
    """
    # format for indexing
    if isinstance(chans, int):
        chans = np.asarray([chans])
    if isinstance(tints, int):
        tints = np.asarray([tints])
    hd = HERAData(zen_path)
    reds = get_reds(hd.antpos, pols=[pol])
    data, flags, _ = hd.read(freq_chans=chans, polarizations=[pol])
    if noise:
        ndata = deepcopy(data)
    # filter bls by bad antennas
    if bad_ants is not None:
        reds = fltBad(reds, bad_ants)
        data = {k: v for k, v in data.items() if not any(i in bad_ants \
                for i in k[:2])}
    redg = groupBls(reds) # baseline grouping

    data = {k: v for k, v in data.items() if k[0] != k[1]} # flt autos
    flags = {k: flags[k] for k in data.keys()} # apply same flt to flags

    # for H3C datasets, where some of the keys are in the wrong order
    # reorder keys such that (i, j, pol), with j>i
    data = dict(sorted({((j, i, p) if i > j else (i, j, p)): v for \
                        (i, j, p), v in data.items()}.items()))

    no_tints = hd.Ntimes
    no_chans = hd.Nfreqs
    if chans is None:
        chans = np.arange(no_chans) # indices, not channel numbers
    if tints is not None:
        # filtering time integrations
        data = {k: v[tints, :] for k, v in data.items()}
        flags = {k: v[tints, :] for k, v in flags.items()}
    else:
        tints = np.arange(no_tints)

    if flag_path is not None:
        # for H1C datasets; H3C+ have different flagging files
        hc = HERACal(flag_path)
        _, cal_flags, _, _ = hc.read()
        # filtering flags data
        cal_flags = {k: v[np.ix_(tints, chans)] for k, v in cal_flags.items()}
        ap1, ap2 = hera_cal.utils.split_pol(pol)
        # updating flags from flag file
        for (i, j, pol) in data.keys():
            flags[(i, j, pol)] += cal_flags[(i, ap1)]
            flags[(i, j, pol)] += cal_flags[(j, ap2)]
        # dict of masked data with updated flags
        data = {k: numpy.ma.array(v, mask=flags[k], fill_value=np.nan) for k, v \
                in data.items()}
        data_size = np.asarray(list(data.values())).flatten().size
        no_flags = np.asarray([d.mask for d in data.values()]).flatten().sum()
        print('{} out of {} data points flagged for visibility dataset {}\n'.\
              format(no_flags, data_size, os.path.basename(zen_path)))

    # Collect data together
    no_tints, no_chans = data[list(data.keys())[0]].shape # get filtered data dimensions
    cdata = numpy.ma.empty((no_chans, no_tints, redg.shape[0]), fill_value=np.nan, \
                           dtype=complex)
    for chan in range(len(chans)):
        cdata[chan, ...] = numpy.ma.hstack([data[(*bl_row[1:], pol)][:, chan, \
                                           np.newaxis] for bl_row in redg])
    if noise:
        # to get correct channel width
        if no_chans == 1:
            chan = chans[0]
            dataf, _, _ = hd.read(freq_chans=np.arange(chan, chan+2), \
                                  polarizations=[pol])
            df = np.ediff1d(dataf.freqs)[0]
        else:
            df = None
        cndata = numpy.empty((no_chans, no_tints, redg.shape[0]), dtype=float)
        ndata = {(*bl, pol): predict_noise_variance_from_autos((*bl, pol), \
                 ndata, df=df) for bl in redg[:, 1:]}
        ndata = {k: v[tints, :] for k, v in ndata.items()}
        for chan in range(len(chans)):
            cndata[chan, ...] = numpy.hstack([ndata[(*bl_row[1:], pol)][:, chan, \
                                             np.newaxis] for bl_row in redg])
        return hd, redg, cdata, cndata
    else:
        return hd, redg, cdata


def redblMap(redg):
    """Return unique baseline types

    :param redg: Grouped baselines, as returned by groupBls
    :type redg: ndarray

    :return: Array of the unique baseline types in the dataset, with the unqiue
    baseline represented by the baseline with the lowest antenna numbers
    :rype: ndarray
    """
    bl_ids = numpy.unique(redg[:, 0], return_index=True)
    return np.array(redg[bl_ids[1], :])


def red_ant_pos(redg, ant_pos):
    """Returns the positions of the antennas that define the redundant sets

    Returns the antenna positions of the baseline that is representative of
    the redundant baseline group it is in (taken to be the baseline with
    the lowest antenna numbers in the group). E.g baseline_id 2 has baselines of
    type (12, 13), therefore we take the positions of antennas (12, 13).

    :param redg: Grouped baselines, as returned by groupBls
    :type redg: ndarray
    :param ant_pos: Antenna positions from HERAData container
    :type ant_pos: dict

    :return: Array of position coordinates of the baselines that define the
    redundant sets. Dimensions are (baseline, (ant1, ant2), coordinates)
    :rtype: ndarray
    """
    red_bl_types = redblMap(redg)
    red_ant_positions = np.array([np.array([ant_pos[i[1]], ant_pos[i[2]]]) \
                       for i in red_bl_types])
    return red_ant_positions


def red_ant_sep(redg, ant_pos):
    """Return seperation of the antennas that define a redundant group

    Seperation defined to be antenna 1 minus antenna 2 in antenna pair

    :param redg: Grouped baselines, as returned by groupBls
    :type redg: ndarray
    :param ant_pos: Antenna positions from HERAData container
    :type ant_pos: dict

    :return: Array of seperations of the baselines that define the redundant sets
    :rtype: ndarray
    """
    red_ant_positions = red_ant_pos(redg, ant_pos)
    red_ant_seperation = red_ant_positions[:, 0, :] - red_ant_positions[:, 1, :]
    return red_ant_seperation


def decomposeCArray(arr):
    """Reformat 1D complex array into 1D real array

    The 1D complex array with elements [z_1, ..., z_i] is reformatted such
    that the new array has elements [Re(z_1), Im(z_1), ..., Re(z_i), Im(z_i)].

    :param arr: Complex array
    :type arr: ndarray

    :return: Real array where the complex elements of arr have been decomposed
    into adjacent real elements
    :rtype: ndarray
    """
    assert arr.ndim == 1
    assert arr.dtype  == numpy.complex
    new_arr = np.vstack((arr.real, arr.imag))
    return new_arr.transpose().flatten()


def makeCArray(arr):
    """Reformat 1D real array of interweaved Re and Im components into 1D
    complex array

    The 1D real array with elements [Re(z_1), Im(z_1), ..., Re(z_n), Im(z_n)]
    is reformatted such that the new array has elements [z_1, ..., z_n], with
    z_i = Re(z_i) + j*Im(z_i).

    :param arr: Real array where the complex elements of a complex array have
    been decomposed into adjacent real elements
    :type arr: ndarray

    :return: Complex array
    :rtype: ndarray
    """
    assert arr.ndim == 1
    assert arr.dtype  == numpy.float
    arr = arr.reshape((-1, 2))
    return arr[:, 0] + 1j*arr[:, 1]


def makeEArray(arr):
    """Reformat 1D real array of interweaved amplitude and phase components into
    1D complex array

    The 1D real array with elements [Amp(z_1), Arg(z_1), ..., Amp(z_n), Arg(z_n)]
    is reformatted such that the new array has elements [z_1, ..., z_n], with
    z_i = Amp(z_i)*exp(j*Arg(z_i)).

    :param arr: Real array where the amplitude and phase components of a complex
    array have been decomposed into adjacent real elements
    :type arr: ndarray

    :return: Complex array
    :rtype: ndarray
    """
    assert arr.ndim == 1
    assert arr.dtype  == numpy.float
    arr = arr.reshape((-1, 2))
    return arr[:, 0] * np.exp(1j*arr[:, 1])


def norm_rel_sols(resx, no_unq_bls, coords='cartesian'):
    """Renormalize relative calibration solutions such that the average gain
    amplitude is 1

    :param resx: Optimization result for the solved antenna gains and true sky
    visibilities
    :type resx: ndarray
    :param no_unq_bls: Number of unique baselines (equivalently the number of
    redundant visibilities)
    :type no_unq_bls: int
    :param coords: Coordinate system in which gain and visibility parameters
    have been set up
    :type coords: str {"cartesian", "polar"}

    :return: Renormalized visibility and gains solutions array
    :rtype:
    """
    vis_params, gain_params = numpy.split(resx, [no_unq_bls*2,])
    if coords == 'polar':
        avg_amp = np.mean(gain_params[::2])
        gain_params[::2] = gain_params[::2] / avg_amp
        vis_params[::2] = vis_params[::2] * avg_amp**2
    elif coords == 'cartesian':
        avg_amp = np.mean(np.abs(makeCArray(gain_params)))
        gain_params = gain_params / avg_amp
        vis_params = vis_params * avg_amp**2
    else:
        raise ValueError('Specify a correct coordinate system: {"cartesian", \
                         "polar"}')
    return np.hstack([vis_params, gain_params])


def flt_ant_pos(ant_pos, ants):
    """Filters antenna positions dictionary from HERAData container by good
    antennas, sorts them, and returns ndarray of shape (no_ants, 3)

    :param ant_pos: Antenna positions from HERAData container
    :type ant_pos: dict
    :param ants: Good antennas
    :type ants: list

    :return: Filtered and sorted antenna positions
    :rtype: ndarray
    """
    flt_ant_pos_dict = dict(sorted({a: p for a, p in ant_pos.items() if a in ants}.items()))
    flt_ant_pos_arr = numpy.asarray(list(flt_ant_pos_dict.values()))
    return flt_ant_pos_arr


def exp_amps(gain_comps):
    """Exponentiate the gain amplitude components of an interweaved ravelled
    array of gain components. Useful if the logarithm of gain amplitudes are
    used as parameters; this forces the gain amplitudes to be positive

    :param gain_comps: Gain component array where log(amplitude) and phase components
    are adjacent
    :type gain_comps: ndarray

    :return: Gain component array where amplitude and phase components are adjacent
    :rtype: ndarray
    """
    gamps = np.exp(gain_comps[::2])
    gphases = gain_comps[1::2]
    gain_comps = np.ravel(np.vstack((gamps, gphases)), order='F')
    return gain_comps


@jit
def gVis(vis, credg, gains):
    """Apply gains to visibilities

    :param vis: visibilities
    :type vis: ndarray
    :param credg: Grouped baselines, condensed so that antennas are
    consecutively labelled. See relabelAnts
    :type credg: ndarray
    :param gains: Antenna gains
    :type gains: ndarray

    :return: Modified visibilities by applying antenna gains
    :rtype: ndarray
    """
    return vis[credg[:, 0]]*gains[credg[:, 1]]*np.conj(gains[credg[:, 2]])


# Negative log-likelihood calculation
NLLFN = {'cauchy': lambda delta: np.log(1 + np.square(np.abs(delta))).sum(),
         'gaussian': lambda delta: np.square(np.abs(delta)).sum(),
         't': lambda dn: (-np.log(gamma(0.5*(dn[1] + 1))) + 0.5*np.log(np.pi*dn[1]) \
         + np.log(gamma(0.5*dn[1])) + 0.5*(dn[1] + 1)*(np.log(1 + (1/dn[1])*np.\
         square(np.abs(dn[0]))))).sum(),
         'cauchy_noise': lambda delta, noise: (np.log(np.pi*noise) + np.log(1 + \
         np.square(np.abs(delta)/noise))).sum(),
         'gaussian_noise': lambda delta, noise: 0.5*(np.log(2*np.pi*noise) + \
         np.square(np.abs(delta)/noise)).sum(),
         't_noise': lambda dn, noise: (-np.log(gamma(0.5*(dn[1] + 1))) + 0.5*np.log(np.pi*dn[1]) \
         + np.log(gamma(0.5*dn[1])) + np.log(noise) + 0.5*(dn[1] + 1)*(np.log(1 + (1/dn[1])*np.\
         square(np.abs(dn[0])/noise)))).sum()
         }

makeC = {'cartesian': makeCArray, 'polar': makeEArray}


def insert_gref(gain_comps, ref_ant_idx):
    """Insert 0 gain phase for reference antenna in gain phase array

    :param gain_comps: Gain component array where amplitude and phase components
    are adjacent
    :type gain_comps: ndarray
    :param ref_ant_idx: Index of reference antenna to set gain phase to 0 - polar
    coordinate system only.
    :type ref_ant_idx: int, None

    :return: Gain component array where amplitude and phase components are adjacent
    :rtype: ndarray
    """
    gamps = gain_comps[::2]
    gphases = gain_comps[1::2]
    gphase1, gphase2 = np.split(gphases, [ref_ant_idx])
    gphaseref = 0. # set the phase of the reference antenna to be 0
    gphases = np.hstack([gphase1, gphaseref, gphase2])
    gain_comps = np.ravel(np.vstack((gamps, gphases)), order='F')
    return gain_comps


def split_rel_results(resx, no_unq_bls, coords='cartesian'):
    """Split the real results array from relative calibration minimization into
    complex visibility and gains arrays

    :param resx: Optimization result for the solved antenna gains and true sky
    visibilities
    :type resx: ndarray
    :param no_unq_bls: Number of unique baselines (equivalently the number of
    redundant visibilities)
    :type no_unq_bls: int
    :param coords: Coordinate system in which gain and visibility parameters
    have been set up
    :type coords: str {"cartesian", "polar"}

    :return: Tuple of visibility and gain solution arrays
    :rtype: tuple
    """
    cfun = {'cartesian':makeCArray, 'polar':makeEArray}
    vis_params, gains_params = numpy.split(resx, [no_unq_bls*2,])
    res_vis = cfun[coords](vis_params)
    res_gains = cfun[coords](gains_params)
    return res_vis, res_gains


def check_ndist(distribution, noise):
    """Check that the correct log-likelihood function is used

    :param distribution: Distribution assumption of noise under MLE {'gaussian',
    'cauchy'}
    :type distribution: str
    :param noise: Noise array to feed into log-likelihood calculations
    :type noise: ndarray

    :return: Corrected noise distribution
    :rtype: str
    """
    if noise is not None:
        if '_noise' not in distribution:
            distribution = distribution + '_noise'
    else:
        if '_noise' in distribution:
            distribution = distribution.replace('_noise', '')
    return distribution


def relative_nlogLkl(credg, distribution, obsvis, no_unq_bls, coords, logamp, \
                     lovamp, tilt_reg, gphase_reg, ant_pos_arr, ref_ant_idx, \
                     phase_reg_initp, noise, params):
    """Redundant relative negative log-likelihood calculator

    We impose that the true sky visibilities from redundant baseline sets are
    equal, and use this prior to calibrate the visibilities (up to some degenerate
    parameters). We set the noise for each visibility to be 1.

    Note: parameter order is such that the function can be usefully partially applied.

    :param credg: Grouped baselines, condensed so that antennas are
    consecutively labelled. See relabelAnts
    :type credg: ndarray
    :param distribution: Distribution assumption of noise under MLE {'gaussian',
    'cauchy'}
    :type distribution: str
    :param obsvis: Observed sky visibilities for a given frequency and given time,
    reformatted to have format consistent with credg
    :type obsvis: ndarray
    :param no_unq_bls: Number of unique baselines (equivalently the number of
    redundant visibilities)
    :type no_unq_bls: int
    :param coords: Coordinate system in which gain and visibility parameters
    have been set up
    :type coords: str {"cartesian", "polar"}
    :param logamp: The logarithm of the amplitude initial parameters is taken,
    such that only positive solutions can be returned. Only if coords=="polar".
    :type logamp: bool
    :param lovamp: The logarithm of the visibility amplitude initial parameters is taken,
    such that only positive solutions can be returned. Only if coords=="polar".
    :type lovamp: bool
    :param tilt_reg: Add regularization term to constrain tilt shifts to 0
    :type tilt_reg: bool
    :param gphase_reg: Add regularization term to constrain the gain phase mean
    :type gphase_reg: bool
    :param ant_pos_arr: Array of filtered antenna position coordinates for the antennas
    in ants. See flt_ant_pos.
    :type ant_pos_arr: ndarray
    :param ref_ant_idx: Index of reference antenna to set gain phase to 0 - polar
    coordinate system only.
    :type ref_ant_idx: int, None
    :param phase_reg_initp: Add regularization term to constrain the phases to be
    the same as the ones from the initial parameters
    :type phase_reg_initp: bool
    :param noise: Noise array to feed into log-likelihood calculations
    :type noise: ndarray
    :param params: Parameters to constrain - redundant visibilities and gains
    (Re & Im [cartesian] or Amp & Phase [polar] components interweaved for both)
    :type params: ndarray

    :return: Negative log-likelihood of MLE computation
    :rtype: float
    """
    vis_comps, gain_comps = np.split(params, [no_unq_bls*2, ])
    if ref_ant_idx is not None:
        gain_comps = insert_gref(gain_comps, ref_ant_idx)
    # transforming gain amplitudes to force positive results
    if logamp:
        gain_comps = exp_amps(gain_comps)
    if lovamp:
        vis_comps = exp_amps(vis_comps)
    vis = makeC[coords](vis_comps)
    gains = makeC[coords](gain_comps)
    delta = obsvis - gVis(vis, credg, gains)
    if 't.' in distribution:
        nu = distribution.split('.')[-1]
        if noise is not None:
            nu = nu.split('_')[0]
            distribution = 't_noise'
        else:
            distribution = 't'
        nu = int(nu)
        delta = delta, nu
    if noise is not None:
        nlog_likelihood = NLLFN[distribution](delta, noise)
    else:
        nlog_likelihood = NLLFN[distribution](delta)
    if tilt_reg or gphase_reg:
        if coords == 'cartesian':
            gphases = np.angle(gains)
        else:
            gphases = gain_comps[1::2]
        if tilt_reg:
            if ant_pos_arr is None:
                raise ValueError('Input antenna position array to constrain tilts')
            tilt_reg_pen = (np.square(np.sum(ant_pos_arr[:, 0]*gphases)) + \
                            np.square(np.sum(ant_pos_arr[:, 1]*gphases)))/(np.pi/180)
            nlog_likelihood += tilt_reg_pen
        if gphase_reg:
            gphase_reg_pen = np.square(gphases.mean())/(np.pi/180)
            nlog_likelihood += gphase_reg_pen
    if phase_reg_initp is not None:
        visr, gainsr = split_rel_results(phase_reg_initp, no_unq_bls, coords=coords)
        deltaa = (np.angle(gainsr) - np.angle(gains) + np.pi) % (2*np.pi) - np.pi
        nlog_likelihood += np.sum(np.square(deltaa))
    return nlog_likelihood


def doRelCal(credg, obsvis, no_unq_bls, no_ants, coords='cartesian', distribution='cauchy', \
             noise=None, bounded=False, logamp=False, lovamp=False, norm_gains=False, \
             tilt_reg=False, gphase_reg=False, ant_pos_arr=None, ref_ant_idx=None, \
             initp=None, max_nit=2000, return_initp=False, jax_minimizer=False, \
             phase_reg_initp=False):
    """Do relative step of redundant calibration

    Initial parameter guesses, if not specified, are 1+1j and 0+0j in cartesian
    coordinates, 1*e^0j and 0*e^0j in polar coordinates for the gains and true
    sky visibilities

    :param credg: Grouped baselines, condensed so that antennas are
    consecutively labelled. See relabelAnts
    :type credg: ndarray
    :param obsvis: Observed sky visibilities for a given frequency and given time,
    reformatted to have format consistent with redg
    :type obsvis: ndarray
    :param no_unq_bls: Number of unique baselines (equivalently the number of
    redundant visibilities)
    :type no_unq_bls: int
    :param no_ants: Number of antennas for given observation
    :type no_ants: int
    :param coords: Coordinate system in which gain and visibility parameters
    have been set up
    :type coords: str {"cartesian", "polar"}
    :param distribution: Distribution assumption of noise under MLE {'gaussian',
    'cauchy'}
    :type distribution: str
    :param noise: Noise array to feed into log-likelihood calculations
    :type noise: ndarray
    :param bounded: Bounded optimization, where the amplitudes for the visibilities
    and the gains must be > 0. 'trust-constr' method used.
    :type bounded: bool
    :param logamp: The logarithm of the gain amplitude initial parameters is taken,
    such that only positive solutions can be returned. Only if coords=="polar".
    :type logamp: bool
    :param lovamp: The logarithm of the visibility amplitude initial parameters is taken,
    such that only positive solutions can be returned. Only if coords=="polar".
    :type lovamp: bool
    :param norm_gains: Normalize result gain amplitudes such that their mean is 1
    :type norm_gains: bool
    :param tilt_reg: Add regularization term to constrain tilt shifts to 0
    :type tilt_reg: bool
    :param gphase_reg: Add regularization term to constrain the gain phase mean
    :type gphase_reg: bool
    :param ant_pos_arr: Array of filtered antenna position coordinates for the antennas
    in ants. See flt_ant_pos.
    :type ant_pos_arr: ndarray
    :param initp: Initial parameter guesses for true visibilities and gains
    :type initp: ndarray, None
    :param ref_ant_idx: Index of reference antenna to set gain phase to 0 - polar
    coordinate system only.
    :type ref_ant_idx: int, None
    :param max_nit: Maximum number of iterations to perform
    :type max_nit: int
    :param return_initp: Return optimization parameters that can be reused
    :type return_initp: bool
    :param jax_minimizer: Use jax minimization implementation - only if unbounded
    :type jax_minimizer: bool
    :param phase_reg_initp: Add regularization term to constrain the phases to be
    the same as the ones from the initial parameters
    :type phase_reg_initp: bool

    :return: Optimization result for the solved antenna gains and true sky
    visibilities
    :rtype: Scipy optimization result object
    """
    if initp is None:
        # set up initial parameters
        if coords == 'cartesian': # (Re & Im components)
            xvis = np.zeros(no_unq_bls*2) # complex vis
            xgains = np.ones(no_ants*2) # complex gains
            if bounded:
                print('Bounds not needed for cartesian coordinate approach')
            if logamp:
                print('logamp method not applicable for cartesian coordinate approach')
            initp = np.hstack([xvis, xgains])
        elif coords == 'polar': # (Amp & Phase components)
            xvphases = np.zeros(no_unq_bls)
            xgphases = np.zeros(no_ants-1)
            if logamp:
                xgamps = np.zeros(no_ants-1)
                add_amp = 0.
                if bounded:
                    print('Disregarding bounded argument in favour of logamp approach')
            else:
                xgamps = np.ones(no_ants-1)
                add_amp = 1.
            if lovamp:
                xvamps = np.repeat(-3., no_unq_bls)
            else:
                xvamps = np.ones(no_unq_bls)
            xvis = np.ravel(np.vstack((xvamps, xvphases)), order='F')
            xgains = np.ravel(np.vstack((xgamps, xgphases)), order='F')
            initp = np.hstack([xvis, xgains])
            initp = np.append(initp, np.array([add_amp])) # add back a gain amplitude
            if ref_ant_idx is None:
                initp = np.append(initp, np.array([0.])) # add back a gphase param
        else:
            raise ValueError('Specify a correct coordinate system: {"cartesian", \
                             "polar"}')

    if phase_reg_initp:
        phase_reg_initp = initp
    else:
        phase_reg_initp = None

    distribution = check_ndist(distribution, noise)
    ff = jit(functools.partial(relative_nlogLkl, credg, distribution, obsvis, \
                               no_unq_bls, coords, logamp, lovamp, tilt_reg, gphase_reg, \
                               ant_pos_arr, ref_ant_idx, phase_reg_initp, noise))

    if noise is not None and distribution != 'gaussian_noise':
        # Convert noise variance to HWHM for Cauchy distribution
        noise = np.sqrt(2*np.log(2))*np.sqrt(noise)

    if tol is None:
        if noise is not None:
            # Increase tol since low noise values greatly increase the fun of
            # minimization
            if distribution != 'gaussian_noise':
                tol = 1e-1
            else:
                tol = 1e4
        else:
            tol = 1e-5 # default for method='BFGS'
        if xd:
            # Increase tol by the number of days for across days rel cal
            tol = tol * obsvis.shape[0]

    if jax_minimizer and not bounded:
        res = jminimize(ff, initp, method='bfgs', tol=tol, \
                        options={'maxiter':max_nit})._asdict()
        print('status: {}'.format(res['status']))
    else:
        if bounded and coords == 'polar' and not logamp:
            lb = numpy.repeat(-np.inf, initp.size)
            ub = numpy.repeat(np.inf, initp.size)
            lb[-2*no_ants::2] = 0 # lower bound for gain amplitudes
            bounds = Bounds(lb, ub)
            method = 'trust-constr'
            hess = jit(jacfwd(jacrev(ff)))
            jac = None
        else:
            bounds = None
            method = 'BFGS'
            jac = jit(jacrev(ff))
            hess = None
        res = minimize(ff, initp, bounds=bounds, method=method, \
                       jac=jac, hess=hess, tol=tol, options={'maxiter':max_nit})
    print(res['message'])
    if return_initp:
        # to reuse parameters
        initp = numpy.copy(res['x'])
    if (logamp or lovamp or ref_ant_idx is not None) and coords == 'polar':
        vis_comps, gain_comps = np.split(res['x'], [no_unq_bls*2, ])
        if ref_ant_idx is not None:
            gain_comps = insert_gref(gain_comps, ref_ant_idx)
        # transforming gain amplitudes back
        if logamp:
            gain_comps = exp_amps(gain_comps)
        if lovamp:
            vis_comps = exp_amps(vis_comps)
        res['x'] = numpy.array(np.hstack([vis_comps, gain_comps]))
    if norm_gains:
        if coords == 'polar' and (res['x'][-2*no_ants::2] < 0).any():
            print('Relative calibration solutions were not normalized, as some '\
                  'negative gain amplitudes were found.')
        else:
            res['x'] = norm_rel_sols(res['x'], no_unq_bls, coords=coords)
    if return_initp:
        retn = res, initp
    else:
        retn = res
    return retn


@jit
def XDgVis(vis, credg, gains):
    """Apply gains to visibilities across days

    :param vis: visibilities
    :type vis: ndarray
    :param credg: Grouped baselines, condensed so that antennas are
    consecutively labelled. See relabelAnts
    :type credg: ndarray
    :param gains: Antenna gains
    :type gains: ndarray

    :return: Modified visibilities by applying antenna gains
    :rtype: ndarray
    """
    return vis[:, credg[:, 0]]*gains[:, credg[:, 1]]*np.conj(gains[:, credg[:, 2]])


def relative_nlogLklD(credg, distribution, obsvis, no_unq_bls, phase_reg_initp, \
                      noise, xd, params):
    """Redundant relative negative log-likelihood calculator

    *DEFAULT IMPLEMENTATION*

    We impose that the true sky visibilities from redundant baseline sets are
    equal, and use this prior to calibrate the visibilities (up to some degenerate
    parameters). We set the noise for each visibility to be 1.

    Note: parameter order is such that the function can be usefully partially applied.

    :param credg: Grouped baselines, condensed so that antennas are
    consecutively labelled. See relabelAnts
    :type credg: ndarray
    :param distribution: Distribution assumption of noise under MLE {'gaussian',
    'cauchy'}
    :type distribution: str
    :param obsvis: Observed sky visibilities for a given frequency and given time,
    reformatted to have format consistent with credg
    :type obsvis: ndarray
    :param no_unq_bls: Number of unique baselines (equivalently the number of
    redundant visibilities)
    :type no_unq_bls: int
    :param phase_reg_initp: Add regularization term to constrain the phases to be
    the same as the ones from the initial parameters
    :type phase_reg_initp: bool
    :param noise: Noise array to feed into log-likelihood calculations
    :type noise: ndarray
    :param xd: Across days calibration
    :type xd: bool
    :param params: Parameters to constrain - redundant visibilities and gains
    (Re & Im components interweaved)
    :type params: ndarray

    :return: Negative log-likelihood of MLE computation
    :rtype: float
    """
    vis_comps, gain_comps = np.split(params, [no_unq_bls*2, ])
    vis = makeCArray(vis_comps)
    gains = makeCArray(gain_comps)
    if xd:
        gains = gains.reshape((obsvis.shape[0], -1))
        vis = np.tile(vis, obsvis.shape[0]).reshape((obsvis.shape[0], -1))
        delta = obsvis - XDgVis(vis, credg, gains)
    else:
        delta = obsvis - gVis(vis, credg, gains)
    if noise is not None:
        nlog_likelihood = NLLFN[distribution](delta, noise)
    else:
        nlog_likelihood = NLLFN[distribution](delta)
    if phase_reg_initp is not None:
        visr, gainsr = split_rel_results(phase_reg_initp, no_unq_bls, coords='cartesian')
        deltaa = (np.angle(gainsr) - np.angle(gains) + np.pi) % (2*np.pi) - np.pi
        nlog_likelihood += np.sum(np.square(deltaa))
    return nlog_likelihood


def doRelCalD(credg, obsvis, no_unq_bls, no_ants, distribution='cauchy',
              noise=None, initp=None, xd=False, return_initp=False, tol=None):
    """Do relative step of redundant calibration

    *DEFAULT IMPLEMENTATION*

    Initial parameter guesses, if not specified, are 1+1j and 0+0j for the gains
    and true sky visibilities.

    :param credg: Grouped baselines, condensed so that antennas are
    consecutively labelled. See relabelAnts
    :type credg: ndarray
    :param obsvis: Observed sky visibilities for a given frequency and given time,
    reformatted to have format consistent with redg
    :type obsvis: ndarray
    :param no_unq_bls: Number of unique baselines (equivalently the number of
    redundant visibilities)
    :type no_unq_bls: int
    :param no_ants: Number of antennas for given observation
    :type no_ants: int
    :param distribution: Distribution assumption of noise under MLE {'gaussian',
    'cauchy'}
    :type distribution: str
    :param noise: Noise array to feed into log-likelihood calculations
    :type noise: ndarray
    :param initp: Initial parameter guesses for true visibilities and gains
    :type initp: ndarray, None
    :param xd: Across days calibration
    :type xd: bool
    :param return_initp: Return optimization parameters that can be reused
    :type return_initp: bool
    :param tol: Set the tolerance for minimization termination
    :type tol: float

    :return: Optimization result for the solved antenna gains and true sky
    visibilities
    :rtype: Scipy optimization result object
    """
    if initp is None:
        # set up initial parameters
        xvis = np.zeros(no_unq_bls*2) # complex vis
        xgains = np.ones(no_ants*2) # complex gains
        if xd:
            xgains = np.tile(xgains, obsvis.shape[0])
        initp = np.hstack([xvis, xgains])
        phase_reg_initp = None
    else:
        if xd:
            # Do not care about phase/tilt regularization
            phase_reg_initp = None
        else:
            phase_reg_initp = initp

    distribution = check_ndist(distribution, noise)

    if noise is not None and distribution != 'gaussian_noise':
        # Convert noise variance to HWHM for Cauchy distribution
        noise = np.sqrt(2*np.log(2))*np.sqrt(noise)

    if tol is None:
        if noise is not None:
            # Increase tol since low noise values greatly increase the fun of
            # minimization
            if distribution != 'gaussian_noise':
                tol = 1e-1
            else:
                tol = 1e4
        else:
            tol = 1e-5 # default for method='BFGS'
        if xd:
            # Increase tol by the number of days for across days rel cal
            tol = tol * obsvis.shape[0]

    ff = jit(functools.partial(relative_nlogLklD, credg, distribution, obsvis, \
                               no_unq_bls, phase_reg_initp, noise, xd))

    res = minimize(ff, initp, bounds=None, method='BFGS', \
                   jac=jit(jacrev(ff)), hess=None, options={'maxiter':2000}, \
                   tol=tol)
    print(res['message'])
    initp = numpy.copy(res['x'])
    res['x'] = norm_rel_sols(res['x'], no_unq_bls, coords='cartesian')
    if return_initp:
        retn = res, initp
    else:
        retn = res
    return retn


def rotate_phase(rel_resx, no_unq_bls, principle_angle=False, norm_gains=False):
    """Rotate phases of gains by +pi and make amplitude positive, for gains that
    have negative amplitude gain solutions from relative redundant calibration

    :param rel_resx: Polar relative calibration results
    :type rel_resx: ndarray
    :param no_unq_bls: Number of unique baselines (equivalently the number of
    redundant visibilities)
    :type no_unq_bls: int
    :param principle_angle:
    :type principle_angle:
    :param norm_gains: Normalize result gain amplitudes such that their mean is 1
    :type norm_gains: bool

    :return: Modified relative calibration results with positive gain amplitudes
    and phases rotated by +pi for the gains with previously negative amplitude
    :rtype: ndarray
    """
    vis_params, gains_params = numpy.split(rel_resx, [no_unq_bls*2,])
    rel_gamps = gains_params[::2]
    rel_gphases = gains_params[1::2]
    rel_vamps = vis_params[::2]
    rel_vphases = vis_params[1::2]
    # rotating phases of gains with negative amplitudes by +pi
    neg_gamp_idxs = numpy.where(rel_gamps < 0)[0]
    rel_gphases[neg_gamp_idxs] += numpy.pi
    # rotating phases of visibilities with negative amplitudes by +pi
    neg_vamp_idxs = numpy.where(rel_vamps < 0)[0]
    rel_vphases[neg_vamp_idxs] += numpy.pi
    if principle_angle:
        # taking the principle angle of those phases (required?)
        rel_gphases[neg_gamp_idxs] = (rel_gphases[neg_gamp_idxs] + numpy.pi) % \
                                     (2*numpy.pi) - numpy.pi
        rel_vphases[neg_vamp_idxs] = (rel_vphases[neg_vamp_idxs] + numpy.pi) % \
                                     (2*numpy.pi) - numpy.pi
    rel_gamps[neg_gamp_idxs] = numpy.abs(rel_gamps[neg_gamp_idxs])
    rel_vamps[neg_vamp_idxs] = numpy.abs(rel_vamps[neg_vamp_idxs])
    mod_gain_params = numpy.ravel(numpy.vstack((rel_gamps, rel_gphases)), \
                                  order='F')
    new_res_relx = numpy.hstack([vis_params, mod_gain_params])
    if norm_gains:
        new_res_relx = norm_rel_sols(new_res_relx, no_unq_bls, coords='polar')
    return new_res_relx


def set_gref(gain_comps, ref_ant_idx, op_ref_ant_idx, constr_phase, amp_constr, \
             logamp):
    """Return gain components with reference gain included, constrained to be
    such that the product of all gain amplitudes is 1 and the mean of all gain
    phases is 0

    :param gain_comps: Interweaved polar gain components without reference gain
    :type gain_comps: ndarray
    :param ref_ant_idx: Index of reference antenna in ordered list of antennas
    :type ref_ant_idx: int
    :param op_ref_ant_idx: Index of reference antenna that sets the overall
    phase to 0 in ordered list of antennas. Default is to not set overall phase.
    If True, selects index 25 (corresponding to antenna 85 in H1C_IDR2 dataset).
    :type op_ref_ant_idx: int
    :param constr_phase: Constrain the phase of the gains, as well as the amplitudes
    :type constr_phase: bool
    :param amp_constr: Constraint to apply to gain amplitudes: either the
    {"prod", "mean"} of gain amplitudes = 1
    :type amp_constr: str
    :param logamp: The logarithm of the amplitude initial parameters is taken,
    such that only positive solutions can be returned. Only if coords=="polar".
    :type logamp: bool

    :return: Interweaved polar gain components with reference gain included
    :rtype: ndarray
    """
    # Cannot assign directly to gain_comps because get
    # TypeError: JAX 'Tracer' objects do not support item assignment
    if op_ref_ant_idx is None:
        if logamp:
            gamps = gain_comps[::2]
            gamps = np.exp(gamps) # transforming gain amplitudes to force positive results
            gphases = gain_comps[1::2]
        else:
            if constr_phase:
                gamps = gain_comps[::2]
                gphases = gain_comps[1::2]
            else:
                gamps = gain_comps[:-1:2]
                gphases1 = gain_comps[1:-1:2]
                gphases2 = gain_comps[-1]
                gphases = np.hstack([gphases1, gphases2])
    else:
        assert op_ref_ant_idx > ref_ant_idx, 'Ensure the index of the '\
        'referance antenna that sets the overall phase is higher than that which sets '\
        'the mean phase'
        if logamp:
            if constr_phase:
                gamps = gain_comps[:-1:2]
                gphases = gain_comps[1:-1:2]
                gamps2 = gain_comps[-1]
                gamps = np.exp(np.hstack([gamps, gamps2]))
            else:
                gamps = gain_comps[:-2:2]
                gphases1 = gain_comps[1:-2:2]
                gamps2 = gain_comps[-2]
                gphases2 = gain_comps[-1]
                gamps = np.hstack([gamps, gamps2])
                gphases = np.hstack([gphases1, gphases2])
        else:
            gamps = gain_comps[::2]
            gphases = gain_comps[1::2]

    if not logamp:
        # set reference gain amplitude constraint
        gamp1, gamp2 = np.split(gamps, [ref_ant_idx])
        if amp_constr == 'prod':
            gampref = 1/gamps.prod() # constraint that the product of amps is 1
        elif amp_constr == 'mean':
            gampref = gamps.size + 1 - gamps.sum() # constraint that the mean of amps is 1
        else:
            raise ValueError('Specify a correct gain amplitude constraint: \
                             {"prod", "mean"}')
        gamps = np.hstack([gamp1, gampref, gamp2])

    if constr_phase:
        # set reference gain phase constraint
        gphase1, gphase2 = np.split(gphases, [ref_ant_idx])
        gphaseref = -gphases.sum() # constraint that the mean phase is 0
        # gphaseref = -circmean(gphases) # constraint that the circular mean phase is 0
        # gphaseref = 0 # set the phase of the reference antenna to be 0
        gphases = np.hstack([gphase1, gphaseref, gphase2])

    if op_ref_ant_idx is not None:
        # set overall gain phase constraint
        gphase1, gphase2 = np.split(gphases, [op_ref_ant_idx])
        gphaseref = 0. # set the phase of the reference antenna to be 0
        gphases = np.hstack([gphase1, gphaseref, gphase2])

    gain_comps = np.ravel(np.vstack((gamps, gphases)), order='F')
    return gain_comps


def relative_nlogLklRP(credg, distribution, obsvis, ref_ant_idx, op_ref_ant_idx, \
                       no_unq_bls, constr_phase, amp_constr, logamp, tilt_reg, gphase_reg, \
                       ant_pos_arr, noise, params):
    """Redundant relative likelihood calculator

    *Polar coordinates with gain constraints*

    We impose that the true sky visibilities from redundant baseline sets are
    equal, and use this prior to calibrate the visibilities (up to some degenerate
    parameters). We set the noise for each visibility to be 1.

    Note: parameter order is such that the function can be usefully partially applied.

    :param credg: Grouped baselines, condensed so that antennas are
    consecutively labelled. See relabelAnts
    :type credg: ndarray
    :param distribution: Distribution assumption of noise under MLE {'gaussian',
    'cauchy'}
    :type distribution: str
    :param obsvis: Observed sky visibilities for a given frequency and given time,
    reformatted to have format consistent with credg
    :type obsvis: ndarray
    :param ref_ant_idx: Index of reference antenna in ordered list of antennas
    :type ref_ant_idx: int
    :param op_ref_ant_idx: Index of reference antenna that sets the overall
    phase to 0 in ordered list of antennas. Default is to not set overall phase.
    If True, selects index 25 (corresponding to antenna 85 in H1C_IDR2 dataset).
    :type op_ref_ant_idx: int
    :param no_unq_bls: Number of unique baselines (equivalently the number of
    redundant visibilities)
    :type no_unq_bls: int
    :param constr_phase: Constrain the phase of the gains, as well as the amplitudes
    :type constr_phase: bool
    :param amp_constr: Constraint to apply to gain amplitudes: either the
    {"prod", "mean"} of gain amplitudes = 1
    :type amp_constr: str
    :param logamp: The logarithm of the amplitude initial parameters is taken,
    such that only positive solutions can be returned. Only if coords=="polar".
    :type logamp: bool
    :param tilt_reg: Add regularization term to constrain tilt shifts to 0
    :type tilt_reg: bool
    :param gphase_reg: Add regularization term to constrain the gain phase mean
    :type gphase_reg: bool
    :param ant_pos_arr: Array of filtered antenna position coordinates for the antennas
    in ants. See flt_ant_pos. Only required for tilt_reg = True.
    :type ant_pos_arr: ndarray
    :param noise: Noise array to feed into log-likelihood calculations
    :type noise: ndarray
    :param params: Parameters to constrain - redundant visibilities and gains
    (Amp & Phase components interweaved for both)
    :type params: ndarray

    :return: Negative log-likelihood of MLE computation
    :rtype: float
    """
    vis_comps, gain_comps = np.split(params, [no_unq_bls*2, ])
    gain_comps = set_gref(gain_comps, ref_ant_idx, op_ref_ant_idx, constr_phase, \
                          amp_constr, logamp)
    gains = makeEArray(gain_comps)
    vis = makeEArray(vis_comps)
    delta = obsvis - gVis(vis, credg, gains)
    if noise is not None:
        nlog_likelihood = NLLFN[distribution](delta, noise)
    else:
        nlog_likelihood = NLLFN[distribution](delta)
    if tilt_reg or gphase_reg:
        gphases = gain_comps[1::2]
        if tilt_reg:
            if ant_pos_arr is None:
                raise ValueError('Input antenna position array to constrain tilts')
            tilt_reg_pen = (np.square(np.sum(ant_pos_arr[:, 0]*gphases)) + \
                            np.square(np.sum(ant_pos_arr[:, 1]*gphases)))/(np.pi/180)
            nlog_likelihood += tilt_reg_pen
        if gphase_reg:
            gphase_reg_pen = np.square(gphases.mean())/(np.pi/180)
            nlog_likelihood += gphase_reg_pen
    return nlog_likelihood


def doRelCalRP(credg, obsvis, no_unq_bls, no_ants, distribution='cauchy', noise=None, \
               ref_ant_idx=16, op_ref_ant_idx=None, constr_phase=False, \
               amp_constr='prod', bounded=False, logamp=False, tilt_reg=False, \
               gphase_reg=False, ant_pos_arr=None, initp=None, max_nit=2000, \
               jax_minimizer=False):
    """Do relative step of redundant calibration

    *Polar coordinates with constraints*

    Initial parameter guesses, if not specified, are 1*e(0j) and 0*e(0j) for
    the gains and the true sky visibilities.

    :param credg: Grouped baselines, condensed so that antennas are
    consecutively labelled. See relabelAnts
    :type credg: ndarray
    :param obsvis: Observed sky visibilities for a given frequency and given time,
    reformatted to have format consistent with redg
    :type obsvis: ndarray
    :param no_unq_bls: Number of unique baselines (equivalently the number of
    redundant visibilities)
    :type no_unq_bls: int
    :param no_ants: Number of antennas for given observation
    :type no_ants: int
    :param refant_idx: Index of reference antenna that sets the average of
    the gain amplitudes and/orphases to 0 in ordered list of antennas.
    Default is 16 (corresponding to antenna 55 in H1C_IDR2 dataset).
    :type ref_ant_idx: int
    :param op_ref_ant_idx: Index of reference antenna that sets the overall
    phase to 0 in ordered list of antennas. Default is to not set overall phase.
    If True, selects index 25 (corresponding to antenna 85 in H1C_IDR2 dataset).
    :type op_ref_ant_idx: int
    :param distribution: Distribution assumption of noise under MLE {'gaussian',
    'cauchy'}
    :type distribution: str
    :param noise: Noise array to feed into log-likelihood calculations
    :type noise: ndarray
    :param constr_phase: Constrain the phase of the gains, as well as the amplitudes
    :type constr_phase: bool
    :param amp_constr: Constraint to apply to gain amplitudes: either the
    {"prod", "mean"} of gain amplitudes = 1
    :type amp_constr: str
    :param bounded: Bounded optimization, where the amplitudes for the visibilities
    and the gains must be > 0. 'trust-constr' method used.
    :type bounded: bool
    :param logamp: The logarithm of the amplitude initial parameters is taken,
    such that only positive solutions can be returned. Only if coords=="polar".
    :type logamp: bool
    :param tilt_reg: Add regularization term to constrain tilt shifts to 0
    :type tilt_reg: bool
    :param gphase_reg: Add regularization term to constrain the gain phase mean
    :type gphase_reg: bool
    :param ant_pos_arr: Array of filtered antenna position coordinates for the antennas
    in ants. See flt_ant_pos. Only required for tilt_reg = True.
    :type ant_pos_arr: ndarray
    :param initp: Initial parameter guesses for true visibilities and gains
    :type initp: ndarray, None
    :param max_nit: Maximum number of iterations to perform
    :type max_nit: int
    :param jax_minimizer: Use jax minimization implementation - only if unbounded
    :type jax_minimizer: bool

    :return: Optimization result for the solved antenna gains and true sky
    visibilities
    :rtype: Scipy optimization result object
    """
    if initp is None:
        # set up initial parameters
        # reference antenna gain components not included in the initial parameters
        xvamps = np.zeros(no_unq_bls) # vis amplitudes
        xvphases = np.zeros(no_unq_bls) # vis phases
        if logamp:
            xgamps = np.zeros(no_ants-2) # gain amps
            xgphases = np.zeros(no_ants-2) # gain phases
            add_amp = 0.
            if bounded:
                print('Disregarding bounded argument in favour of logamp approach')
            if amp_constr == 'prod':
                print('Constraining the mean of gain amplitudes to be 1 instead '\
                      'of the product, when logamp is True')
        else:
            xgamps = np.ones(no_ants-2) # gain amps
            xgphases = np.zeros(no_ants-2) # gain phases
            add_amp = 1.
        xvis = np.ravel(np.vstack((xvamps, xvphases)), order='F')
        xgains = np.ravel(np.vstack((xgamps, xgphases)), order='F')
        initp = np.hstack([xvis, xgains])
        initp = np.append(initp, np.array([add_amp])) # add back a gain amplitude
        if op_ref_ant_idx is None:
            # overall phase constraint (set reference antenna gain phase to 0)
            initp = np.append(initp, np.array([0.])) # add back a gphase param
        if logamp:
            # log(amp) method
            initp = np.append(initp, np.array([add_amp])) # add back a gamp param
        if not constr_phase:
            # mean phase
            initp = np.append(initp, np.array([0.])) # add back a gphase param

    if type(op_ref_ant_idx) == bool and op_ref_ant_idx:
        op_ref_ant_idx = 25 # suitable antenna for H1C_IDR2

    distribution = check_ndist(distribution, noise)
    ff = jit(functools.partial(relative_nlogLklRP, credg, distribution, obsvis, \
             ref_ant_idx, op_ref_ant_idx, no_unq_bls, constr_phase, amp_constr, \
             logamp, tilt_reg, gphase_reg, ant_pos_arr, noise))

    if noise is not None:
        # Since low noise values greatly increase the function being minimized
        if distribution != 'gaussian_noise':
            # Convert noise variance to HWHM for cauchy distribution
            noise = np.sqrt(2*np.log(2))*np.sqrt(noise)
            tol = 1e-1
        else:
            tol = 1e3
    else:
        tol = None
    if jax_minimizer and not bounded:
        res = jminimize(ff, initp, method='bfgs', tol=tol, \
                        options={'maxiter':max_nit})._asdict()
        print('status: {}'.format(res['status']))
    else:
        if bounded and not logamp:
            lb = numpy.repeat(-np.inf, initp.size)
            ub = numpy.repeat(np.inf, initp.size)
            lb[:2*(no_unq_bls+no_ants-1):2] = 0 # lower bound for gain and vis amps
            bounds = Bounds(lb, ub)
            method = 'trust-constr'
            jac = None
            hess = jit(jacfwd(jacrev(ff)))
        else:
            bounds = None
            method = 'BFGS'
            jac = jit(jacrev(ff))
            hess = None
        res = minimize(ff, initp, bounds=bounds, method=method, \
                       jac=jac, hess=hess, tol=tol, options={'maxiter':max_nit})
    print(res['message'])
    initp = numpy.copy(res['x']) # to reuse parameters
    vis_comps, gain_comps = np.split(res['x'], [no_unq_bls*2, ])
    res['x'] = np.hstack([vis_comps, set_gref(gain_comps, ref_ant_idx, op_ref_ant_idx, \
                                              constr_phase, amp_constr, logamp)])
    if logamp:
        res['x'] = norm_rel_sols(numpy.array(res['x']), no_unq_bls, coords='polar')
    return res, initp


@jit
def degVis(ant_sep, rel_vis, amp, phase_grad_x, phase_grad_y):
    """Transform redundant visibilities according to the degenerate redundant
    parameters

    :param ant_sep: Antenna seperation for baseline types
    :type ant_sep: ndarray
    :param rel_vis: Visibility solutions for redundant baseline groups after
    relative calibration
    :param rel_vis: ndarray
    :param amp: Overall amplitude
    :type amp: float
    :param phase_grad_x: Phase gradient component in x-direction
    :type phase_grad_x: float
    :param phase_grad_y: Phase gradient component in y-direction
    :type phase_grad_y: float

    :return: Transformed relatively calibrated true visibilities
    :rtype: ndarray
    """
    x_sep = ant_sep[:, 0]
    y_sep = ant_sep[:, 1]
    w_alpha = amp**2 * np.exp(1j * (phase_grad_x * x_sep + phase_grad_y \
               * y_sep)) * rel_vis
    return w_alpha


def optimal_nlogLkl(credg, distribution, ant_sep, obsvis, rel_vis, no_ants, \
                    logamp, params):
    """Optimal negative log-likelihood calculator

    We solve for the degeneracies in redundant calibration. This must be done
    after relative redundant calibration. We also set the noise for each visibility
    to be 1.

    :param credg: Grouped baselines, condensed so that antennas are
    consecutively labelled. See relabelAnts
    :type credg: ndarray
    :param distribution: Distribution assumption of noise under MLE {'gaussian',
    'cauchy'}
    :type distribution: str
    :param obsvis: Observed sky visibilities for a given frequency and given time,
    reformatted to have format consistent with credg
    :type obsvis: ndarray
    :param ant_sep: Antenna seperation for baseline types
    :type ant_sep: ndarray
    :param rel_vis: Visibility solutions for redundant baseline groups after
    relative calibration
    :param rel_vis: ndarray
    :param logamp: The logarithm of the amplitude initial parameters is taken,
    such that only positive solutions can be returned
    :type logamp: bool
    :param params: Parameters to constrain: normalized gains (amp and phase
    components interweaved), overall amplitude, overall phase and phase
    gradients in x and y
    :type params: ndarray

    :return: Negative log-likelihood of MLE computation
    :rtype: float
     """
    rel_gain_comps, deg_params = np.split(params, [2*no_ants,])
    if logamp:
        rel_gain_comps = exp_amps(rel_gain_comps)
        o_amp, deg_rest = np.split(deg_params, [1,])
        deg_params = np.hstack([np.exp(o_amp), deg_rest])
    rel_gains = makeEArray(rel_gain_comps)
    w_alpha = degVis(ant_sep, rel_vis, *deg_params[numpy.asarray([0, 2, 3])])
    delta = obsvis - gVis(w_alpha, credg, rel_gains)
    nlog_likelihood = NLLFN[distribution](delta)
    return nlog_likelihood


class Opt_Constraints:
    """Gain, phase and phase gradient constraints for optimal redundant
    calibration

    Parameters to feed into constraint functions must be a flattened real array
    of normalized gains, overall_amplitude, overall phase, phase gradient x and
    phase gradient y, in tha order.

    :param no_ants: Number of antennas for given observation
    :type no_ants: int
    :param ref_ant_idx: Index of reference antenna in ordered list of antennas to
    constrain overall phase. Default is 16 (corresponding to antenna 55 in H1C_IDR2
    dataset).
    :type ref_ant_idx: int
    :param ant_pos_arr: Array of filtered antenna position coordinates for the antennas
    in ants. See flt_ant_pos.
    :type ant_pos_arr: ndarray
    :param logamp: The logarithm of the amplitude initial parameters is taken,
    such that only positive solutions can be returned
    :type logamp: bool
    :param params: Parameters to feed into optimal absolute calibration
    :type params: ndarray
    """
    def __init__(self, no_ants, ref_ant_idx, ant_pos_arr, logamp):
        self.no_ants = no_ants
        self.ref_ant_idx = ref_ant_idx
        self.ant_pos_arr = ant_pos_arr
        self.x_pos = ant_pos_arr[:, 0]
        self.y_pos = ant_pos_arr[:, 1]
        self.logamp = logamp

    def avg_amp(self, params):
        """Constraint that mean of gain amplitudes must be equal to 1

        :return: Residual between mean gain amplitudes and 1
        :rtype: float
        """
        amps = params[:self.no_ants*2:2]
        if self.logamp:
            amps = np.exp(amps)
        return np.mean(amps) - 1

    def avg_phase(self, params):
        """Constraint that mean of gain phases must be equal to 0

        :return: Residual between mean of gain phases and 0
        :rtype: float
        """
        phases = params[1:self.no_ants*2:2]
        return np.mean(phases)

    def ref_phase(self, params):
        """Set phase of reference antenna gain to 0 to set overall phase

        :return: Residual between reference antenna phase and 0
        :rtype: float
        """
        phases = params[1:self.no_ants*2:2]
        return phases[self.ref_ant_idx]

    def ref_amp(self, params):
        """Set amplitude of reference antenna gain to 1 to set overall amplitude

        :return: Residual between reference antenna amplitude and 1
        :rtype: float
        """
        amps = params[:self.no_ants*2:2]
        return amps[self.ref_ant_idx] - 1

    def phase_grad_x(self, params):
        """Constraint that phase gradient in x is 0

        :return: Residual between phase gradient in x and 0
        :rtype: float
        """
        phases = params[1:self.no_ants*2:2]
        return np.sum(phases*self.x_pos)

    def phase_grad_y(self, params):
        """Constraint that phase gradient in y is 0

        :return: Residual between phase gradient in y and 0
        :rtype: float
        """
        phases = params[1:self.no_ants*2:2]
        return np.sum(phases*self.y_pos)


def doOptCal(credg, obsvis, no_ants, ant_pos_arr, ant_sep, rel_vis, distribution='cauchy', \
             ref_ant_idx=16, logamp=False, initp=None, max_nit=1000):
    """Do optimal absolute step of redundant calibration

    Initial degenerate parameter guesses are 0 for the overall amplitude,
    overall phase and the phase gradients in x and y.

    :param credg: Grouped baselines, condensed so that antennas are
    consecutively labelled. See relabelAnts
    :type credg: ndarray§
    :param obsvis: Observed sky visibilities for a given frequency and given time,
    reformatted to have format consistent with redg
    :type obsvis: ndarray
    :param no_ants: Number of antennas for given observation
    :type no_ants: int
    :param ant_pos_arr: Array of filtered antenna position coordinates for the antennas
    in ants. See flt_ant_pos.
    :type ant_pos_arr: ndarray
    :param ant_sep: Antenna seperation for baseline types. See red_ant_sep.
    :type ant_sep: ndarray
    :param rel_vis: Visibility solutions for redundant baseline groups after
    relative calibration
    :param rel_vis: ndarray
    :param distribution: Distribution assumption of noise under MLE {'gaussian',
    'cauchy'}
    :type distribution: str
    :param ref_ant_idx: Index of reference antenna in ordered list of antennas to
    constrain overall phase. Default is 16 (corresponding to antenna 55 in H1C_IDR2
    dataset).
    :type ref_ant_idx: int
    :param logamp: The logarithm of the amplitude initial parameters is taken,
    such that only positive solutions can be returned
    :type logamp: bool
    :param initp: Initial parameter guesses for gains and degenerate parameters
    :type initp: ndarray, None
    :param max_nit: Maximum number of iterations to perform
    :type max_nit: int

    :return: Optimization result for the optimally absolutely calibrated
    redundant gains and degenerate parameters
    :rtype: Scipy optimization result object
    """
    if initp is None:
        # set up initial parameters
        if logamp:
            xgamps = np.zeros(no_ants) # gain amplitudes
            xdegparams = np.zeros(4) # overall amplitude, tilts
        else:
            xgamps = np.ones(no_ants) # gain amplitudes
            xdegparams = np.array([1., 0., 0., 0.]) # overall amplitude, tilts
        xgphases = np.zeros(no_ants) # gain phases
        xgains = np.ravel(np.vstack((xgamps, xgphases)), order='F')
        initp = np.hstack([xgains, xdegparams])

    lb = numpy.repeat(-np.inf, initp.size)
    ub = numpy.repeat(np.inf, initp.size)
    if not logamp:
        lb[:no_ants*2:2] = 0 # lower bound for gain amplitudes
        lb[-4] = 0 # lower bound for overall amplitude
        bounds = Bounds(lb, ub)
    else:
        bounds = None
        print('Disregarding bounds on amplitudes as using logamp method')

    # constraints for optimization
    constraints = Opt_Constraints(no_ants, ref_ant_idx, ant_pos_arr, logamp)
    cons = [
            {'type': 'eq', 'fun': constraints.avg_amp},
            {'type': 'eq', 'fun': constraints.avg_phase},
            {'type': 'eq', 'fun': constraints.ref_phase},
            # {'type': 'eq', 'fun': constraints.ref_amp}, # not needed (?)
            {'type': 'eq', 'fun': constraints.phase_grad_x},
            {'type': 'eq', 'fun': constraints.phase_grad_y}
            ]

    ff = jit(functools.partial(optimal_nlogLkl, credg, distribution, \
                               ant_sep, obsvis, rel_vis, no_ants, logamp))
    res = minimize(ff, initp, jac=jit(jacrev(ff)), hess=jit(jacfwd(jacrev(ff))), \
                   constraints=cons, bounds=bounds, method='trust-constr', \
                   options={'maxiter':max_nit})
    print(res['message'])
    if logamp:
        rel_gain_comps, deg_params = np.split(res['x'], [2*no_ants,])
        rel_gain_comps = exp_amps(rel_gain_comps)
        o_amp, deg_rest = np.split(deg_params, [1,])
        deg_params = np.hstack([np.exp(o_amp), deg_rest])
        res['x'] = np.hstack([rel_gain_comps, deg_params])
    return res


def deg_nlogLkl(distribution, ant_sep, rel_vis1, rel_vis2, params):
    """Degenerate negative log-likelihood calculator

    Max-likelihood estimate to solve for the degenerate parameters that transform
    between the visibility solutions of two different datasets after relative
    calibration

    :param distribution: Distribution assumption of noise under MLE {'gaussian',
    'cauchy'}
    :type distribution: str
    :param ant_sep: Antenna seperation for baseline types
    :type ant_sep: ndarray
    :param rel_vis1: Visibility solutions for redundant baseline groups after
    relative calibration for dataset 1
    :type rel_vis1: ndarray
    :param rel_vis2: Visibility solutions for redundant baseline groups after
    relative calibration for dataset 2
    :type rel_vis2: ndarray
    :param params: Parameters to constrain: overall amplitude, overall phase and
    phase gradients in x and y
    :type params: ndarray

    :return: Negative log-likelihood of MLE computation
    :rtype: float
    """
    w_alpha = degVis(ant_sep, rel_vis1, *params)
    delta = rel_vis2 - w_alpha
    nlog_likelihood = NLLFN[distribution](delta)
    return nlog_likelihood


def doDegVisVis(ant_sep, rel_vis1, rel_vis2, distribution='cauchy', \
                initp=None, max_nit=1000):
    """
    Fit degenerate redundant calibration parameters so that rel_vis1 is as
    close to as possible to rel_vis1

    :param ant_sep: Antenna seperation for baseline types
    :type ant_sep: ndarray
    :param rel_vis1: Visibility solutions for observation set 1 for redundant
    baseline groups after relative calibration
    :param rel_vis1: ndarray
    :param rel_vis2: Visibility solutions for observation set 2 for redundant
    baseline groups after relative calibration
    :param rel_vis2: ndarray
    :param distribution: Distribution assumption of noise under MLE {'gaussian',
    'cauchy'}
    :type distribution: str
    :param initp: Initial parameter guesses for degenerate parameters (overall
    amplitude, x phase gradient and y phase gradient)
    :type initp: ndarray, None
    :param max_nit: Maximum number of iterations to perform
    :type max_nit: int

    :return: Optimization result for the solved degenerate parameters that
    translat between the two datasets
    :rtype: Scipy optimization result object
    """
    if initp is None:
        # set up initial params: overall amp, x and y phase gradients
        initp = np.asarray([1, 0, 0])

    ff = jit(functools.partial(deg_nlogLkl, distribution, ant_sep, \
                               rel_vis1, rel_vis2))
    res = minimize(ff, initp, jac=jit(jacrev(ff)), options={'maxiter':max_nit})
    print(res['message'])
    return res
