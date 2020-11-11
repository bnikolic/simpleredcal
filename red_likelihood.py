"""Robust redundant calibration"""


import os
import functools

import numpy
from matplotlib import pyplot as plt
from scipy.optimize import Bounds, minimize
from scipy.stats import circmean

import hera_cal
from hera_cal.io import HERACal, HERAData
from hera_cal.redcal import get_reds

from jax.config import config
config.update('jax_enable_x64', True)
import jax
from jax import jit, jacrev, jacfwd
from jax.scipy.optimize import minimize as jminimize

# n.b. where 'numpy' is used below it has to be real numpy. 'np' can be
# either jax or real numpy
np=jax.numpy


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
               flag_path=None):
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

    :return hd: HERAData class
    :rtype hd: HERAData class
    :return redg: Grouped baselines, as returned by groupBls
    :rtype redg: ndarray
    :return cdata: Grouped visibilities with flags in numpy MaskedArray format,
    with format consistent with redg and dimensions (freq chans,
    time integrations, baselines)
    :rtype cdata: MaskedArray
    """
    # format for indexing
    if isinstance(chans, int):
        chans = np.asarray([chans])
    if isinstance(tints, int):
        tints = np.asarray([tints])
    hd = HERAData(zen_path)
    reds = get_reds(hd.antpos, pols=[pol])
    data, flags, _ = hd.read(freq_chans=chans, polarizations=[pol])
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

    no_tints, no_chans = data[list(data.keys())[0]].shape # get data dimensions
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

    :param ant_pos: Dictionary of antenna position coordinates for the antennas
    in ants
    :type ant_pos: dict
    :param ants: Good antennas
    :type ants: list

    :return: Filtered and sorted antenna positions
    :rtype: ndarray
    """
    flt_ant_pos_dict = dict(sorted({a: p for a, p in ant_pos.items() if a in ants}.items()))
    flt_ant_pos_arr = numpy.asarray(list(flt_ant_pos_dict.values()))
    return flt_ant_pos_arr


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


LLFN = {'cauchy':lambda delta: np.log(1 + np.square(np.abs(delta))).sum(),
        'gaussian':lambda delta: np.square(np.abs(delta)).sum()}

makeC = {'cartesian': makeCArray, 'polar': makeEArray}


def relative_logLkl(credg, distribution, obsvis, no_unq_bls, coords, params):
    """Redundant relative likelihood calculator

    We impose that the true sky visibilities from redundant baseline sets are
    equal, and use this prior to calibrate the visibilities (up to some degenerate
    parameters). We set the noise for each visibility to be 1.

    Note: parameter order is such that the function can be usefully partially applied.

    :param credg: Grouped baselines, condensed so that antennas are
    consecutively labelled. See relabelAnts
    :type credg: ndarray
    :param distribution: Distribution to fit likelihood {'gaussian', 'cauchy'}
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
    :param params: Parameters to constrain - redundant visibilities and gains
    (Re & Im [cartesian] or Amp & Phase [polar] components interweaved for both)
    :type params: ndarray

    :return: Negative log-likelihood of MLE computation
    :rtype: float
    """
    vis_comps, gains_comps = np.split(params, [no_unq_bls*2, ])
    vis = makeC[coords](vis_comps)
    gains = makeC[coords](gains_comps)

    delta = obsvis - gVis(vis, credg, gains)
    log_likelihood = LLFN[distribution](delta)
    return log_likelihood


def doRelCal(credg, obsvis, no_unq_bls, no_ants, coords='cartesian', distribution='cauchy', \
             bounded=False, initp=None, norm_gains=False, max_nit=1000, \
             jax_minimizer=False):
    """Do relative step of redundant calibration

    Initial parameter guesses, if not specified, are 1+1j for both the gains
    and the true sky visibilities.

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
    :param ref_ant_idx: Index of reference antenna in ordered list of antennas.
    Default is 16 (corresponding to antenna 55 in H1C_IDR2 dataset).
    :type ref_ant_idx: int
    :param coords: Coordinate system in which gain and visibility parameters
    have been set up
    :type coords: str {"cartesian", "polar"}
    :param distribution: Distribution to fit likelihood {'gaussian', 'cauchy'}
    :type distribution: str
    :param bounded: Bounded optimization, where the amplitudes for the visibilities
    and the gains must be > 0. 'trust-constr' method used.
    :type bounded: bool
    :param initp: Initial parameter guesses for true visibilities and gains
    :type initp: ndarray, None
    :param norm_gains: Normalize result gain amplitudes such that their mean is 1
    :type norm_gains: bool
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
        if coords == 'cartesian': # (Re & Im components)
            xvis = np.zeros(no_unq_bls*2) # complex vis
            xgains = np.ones(no_ants*2) # complex gains
        elif coords == 'polar': # (Amp & Phase components)
            xvamps = np.zeros(no_unq_bls) # vis amplitudes
            xvphases = np.zeros(no_unq_bls) # vis phases
            xgamps = np.ones(no_ants) # gain amplitudes
            xgphases = np.zeros(no_ants) # gain phases
            xvis = np.ravel(np.vstack((xvamps, xvphases)), order='F')
            xgains = np.ravel(np.vstack((xgamps, xgphases)), order='F')
        else:
            raise ValueError('Specify a correct coordinate system: {"cartesian", \
                             "polar"}')
        initp = np.hstack([xvis, xgains])

    ff = jit(functools.partial(relative_logLkl, credg, distribution, obsvis, \
                               no_unq_bls, coords))

    if jax_minimizer and not bounded:
        res = jminimize(ff, initp, method='bfgs', options={'maxiter':max_nit})\
              ._asdict()
        print('status: {}'.format(res['status']))
    else:
        if bounded and coords == 'polar':
            lb = numpy.repeat(-np.inf, initp.size)
            ub = numpy.repeat(np.inf, initp.size)
            lb[::2] = 0 # lower bound for gain and vis amplitudes
            bounds = Bounds(lb, ub)
            # method = 'L-BFGS-B' # some gain amplitudes = 0 with this method...
            # hess = None
            # jac = lambda x: numpy.array(jacrev(ff)(x))
            # max_nit = max(10000, max_nit)
            method = 'trust-constr'
            hess = jacfwd(jacrev(ff))
            jac = None
        else:
            bounds = None
            method = 'BFGS'
            jac = jacrev(ff)
            hess = None
        res = minimize(ff, initp, bounds=bounds, method=method, \
                       jac=jac, hess=hess, options={'maxiter':max_nit})
    print(res['message'])
    if norm_gains:
        if coords == 'polar' and (res['x'][-2*no_ants::2] < 0).any():
            print('Relative calibration solutions were not normalized, as some '\
                  'negative gain amplitudes were found.')
        else:
            res['x'] = norm_rel_sols(res['x'], no_unq_bls, coords=coords)
    return res


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


def set_gref(gain_comps, ref_ant_idx, constr_phase, amp_constr):
    """Return gain components with reference gain included, constrained to be
    such that the product of all gain amplitudes is 1 and the mean of all gain
    phases is 0

    :param gain_comps: Interweaved polar gain components without reference gain
    :type gain_comps: ndarray
    :param ref_ant_idx: Index of reference antenna in ordered list of antennas
    :type ref_ant_idx: int
    :param constr_phase: Constrain the phase of the gains, as well as the amplitudes
    :type constr_phase: bool
    :param amp_constr: Constraint to apply to gain amplitudes: either the
    {"prod", "mean"} of gain amplitudes = 1
    :type amp_constr: str

    :return: Interweaved polar gain components with reference gain included
    :rtype: ndarray
    """
    # Cannot assign directly to gain_comps because get
    # TypeError: JAX 'Tracer' objects do not support item assignment
    if constr_phase:
        gamps = gain_comps[::2]
    else:
        gamps = gain_comps[:-1:2]
        gphaseref = gain_comps[-1]
    gphases = gain_comps[1::2]
    gamp1, gamp2 = np.split(gamps, [ref_ant_idx])
    gphase1, gphase2 = np.split(gphases, [ref_ant_idx])
    # set reference gain constraints
    if amp_constr == 'prod':
        gampref = 1/gamps.prod() # constraint that the product of amps is 1
    elif amp_constr == 'mean':
        gampref = 1 + gamps.size - gamps.sum() # constraint that the sum of amps is 1
    else:
        raise ValueError('Specify a correct gain amplitude constraint: \
                         {"prod", "mean"}')
    if constr_phase:
        gphaseref = -gphases.sum() # constraint that the mean phase is 0
        # gphaseref = -circmean(gphases) # constraint that the circular mean phase is 0
        # gphaseref = 0 # set the phase of the reference antenna to be 0
    gamps = np.hstack([gamp1, gampref, gamp2])
    gphases = np.hstack([gphase1, gphaseref, gphase2])
    gain_comps = np.ravel(np.vstack((gamps, gphases)), order='F')
    return gain_comps


def relative_logLklRP(credg, distribution, obsvis, ref_ant_idx, no_unq_bls, \
                      constr_phase, amp_constr, params):
    """Redundant relative likelihood calculator

    *Polar coordinates with gain constraints*

    We impose that the true sky visibilities from redundant baseline sets are
    equal, and use this prior to calibrate the visibilities (up to some degenerate
    parameters). We set the noise for each visibility to be 1.

    Note: parameter order is such that the function can be usefully partially applied.

    :param credg: Grouped baselines, condensed so that antennas are
    consecutively labelled. See relabelAnts
    :type credg: ndarray
    :param distribution: Distribution to fit likelihood {'gaussian', 'cauchy'}
    :type distribution: str
    :param obsvis: Observed sky visibilities for a given frequency and given time,
    reformatted to have format consistent with credg
    :type obsvis: ndarray
    :param ref_ant_idx: Index of reference antenna in ordered list of antennas
    :type ref_ant_idx: int
    :param no_unq_bls: Number of unique baselines (equivalently the number of
    redundant visibilities)
    :type no_unq_bls: int
    :param constr_phase: Constrain the phase of the gains, as well as the amplitudes
    :type constr_phase: bool
    :param amp_constr: Constraint to apply to gain amplitudes: either the
    {"prod", "mean"} of gain amplitudes = 1
    :type amp_constr: str
    :param params: Parameters to constrain - redundant visibilities and gains
    (Amp & Phase components interweaved for both)
    :type params: ndarray

    :return: Negative log-likelihood of MLE computation
    :rtype: float
    """
    vis_comps, gain_comps = np.split(params, [no_unq_bls*2, ])

    vis = makeEArray(vis_comps)
    gains = makeEArray(set_gref(gain_comps, ref_ant_idx, constr_phase, \
                                amp_constr))

    delta = obsvis - gVis(vis, credg, gains)
    log_likelihood = LLFN[distribution](delta)
    return log_likelihood


def doRelCalRP(credg, obsvis, no_unq_bls, no_ants, distribution='cauchy', ref_ant_idx=16, \
               constr_phase=False, amp_constr='prod', bounded=False, initp=None, \
               max_nit=1000, jax_minimizer=False):
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
    :param ref_ant_idx: Index of reference antenna in ordered list of antennas.
    Default is 16 (corresponding to antenna 55 in H1C_IDR2 dataset).
    :type ref_ant_idx: int
    :param distribution: Distribution to fit likelihood {'gaussian', 'cauchy'}
    :type distribution: str
    :param constr_phase: Constrain the phase of the gains, as well as the amplitudes
    :type constr_phase: bool
    :param amp_constr: Constraint to apply to gain amplitudes: either the
    {"prod", "mean"} of gain amplitudes = 1
    :type amp_constr: str
    :param bounded: Bounded optimization, where the amplitudes for the visibilities
    and the gains must be > 0. 'trust-constr' method used.
    :type bounded: bool
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
        # reference antenna gain not included in the initial parameters
        xvamps = np.zeros(no_unq_bls) # vis amplitudes
        xvphases = np.zeros(no_unq_bls) # vis phases
        xgamps = np.ones(no_ants-1) # gain amplitudes
        xgphases = np.zeros(no_ants-1) # gain phases
        xvis = np.ravel(np.vstack((xvamps, xvphases)), order='F')
        xgains = np.ravel(np.vstack((xgamps, xgphases)), order='F')
        initp = np.hstack([xvis, xgains])
        if not constr_phase:
            initp = np.append(initp, np.array([0]))

    ff = jit(functools.partial(relative_logLklRP, credg, \
             distribution, obsvis, ref_ant_idx, no_unq_bls, constr_phase, \
             amp_constr))

    if jax_minimizer and not bounded:
        res = jminimize(ff, initp, method='bfgs', options={'maxiter':max_nit})\
              ._asdict()
        print('status: {}'.format(res['status']))
    else:
        if bounded:
            lb = numpy.repeat(-np.inf, initp.size)
            ub = numpy.repeat(np.inf, initp.size)
            lb[:2*(no_unq_bls+no_ants-1):2] = 0 # lower bound for gain and vis amplitudes
            bounds = Bounds(lb, ub)
            # method = 'L-BFGS-B' # get b'ABNORMAL_TERMINATION_IN_LNSRCH'
            # jac = lambda x: numpy.array(jacrev(ff)(x))
            # hess = None
            # max_nit = min(15000, max_nit)
            method = 'trust-constr'
            jac = None
            hess = jacfwd(jacrev(ff))
        else:
            bounds = None
            method = 'BFGS'
            jac = jacrev(ff)
            hess = None
        res = minimize(ff, initp, bounds=bounds, method=method, \
                       jac=jac, hess=hess, options={'maxiter':max_nit})
    print(res['message'])
    initp = numpy.copy(res['x']) # to reuse parameters
    vis_comps, gain_comps = np.split(res['x'], [no_unq_bls*2, ])
    res['x'] = np.hstack([vis_comps, set_gref(gain_comps, ref_ant_idx, \
                                              constr_phase, amp_constr)])
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


def optimal_logLkl(credg, distribution, ant_sep, obsvis, rel_vis, no_ants, params):
    """Optimal likelihood calculator

    We solve for the degeneracies in redundant calibration. This must be done
    after relative redundant calibration. We also set the noise for each visibility
    to be 1.

    :param credg: Grouped baselines, condensed so that antennas are
    consecutively labelled. See relabelAnts
    :type credg: ndarray
    :param distribution: Distribution to fit likelihood {'gaussian', 'cauchy'}
    :type distribution: str
    :param obsvis: Observed sky visibilities for a given frequency and given time,
    reformatted to have format consistent with credg
    :type obsvis: ndarray
    :param ant_sep: Antenna seperation for baseline types
    :type ant_sep: ndarray
    :param rel_vis: Visibility solutions for redundant baseline groups after
    relative calibration
    :param rel_vis: ndarray
    :param params: Parameters to constrain: normalized gains (amp and phase
    components interweaved), overall amplitude, overall phase and phase
    gradients in x and y
    :type params: ndarray

    :return: Negative log-likelihood of MLE computation
    :rtype: float
     """
    rel_gains_comps, deg_params = np.split(params, [2*no_ants,])
    rel_gains = makeEArray(rel_gains_comps)

    w_alpha = degVis(ant_sep, rel_vis, *deg_params[[0, 2, 3]])
    delta = obsvis - gVis(w_alpha, credg, rel_gains)
    log_likelihood = LLFN[distribution](delta)
    return log_likelihood


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
    :param ant_pos: Array of filtered antenna position coordinates for the antennas
    in ants. See flt_ant_pos.
    :type ant_pos: ndarray
    :param params: Parameters to feed into optimal absolute calibration
    :type params: ndarray
    """
    def __init__(self, no_ants, ref_ant_idx, ant_pos):
        self.no_ants = no_ants
        self.ref_ant_idx = ref_ant_idx
        self.ant_pos = ant_pos
        self.x_pos = ant_pos[:, 0]
        self.y_pos = ant_pos[:, 1]

    def avg_amp(self, params):
        """Constraint that mean of gain amplitudes must be equal to 1

        :return: Residual between mean gain amplitudes and 1
        :rtype: float
        """
        amps = params[:self.no_ants*2:2]
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


def doOptCal(credg, obsvis, no_ants, ant_pos, ant_sep, rel_vis, distribution='cauchy', \
             ref_ant_idx=16, initp=None, max_nit=1000):
    """Do optimal absolute step of redundant calibration

    Initial degenerate parameter guesses are 1 for the overall amplitude, and 0
    for the overall phase and the phase gradients in x and y.

    :param credg: Grouped baselines, condensed so that antennas are
    consecutively labelled. See relabelAnts
    :type credg: ndarray
    :param obsvis: Observed sky visibilities for a given frequency and given time,
    reformatted to have format consistent with redg
    :type obsvis: ndarray
    :param no_ants: Number of antennas for given observation
    :type no_ants: int
    :param ant_pos: Array of filtered antenna position coordinates for the antennas
    in ants. See flt_ant_pos.
    :type ant_pos: ndarray
    :param ant_sep: Antenna seperation for baseline types. See red_ant_sep.
    :type ant_sep: ndarray
    :param rel_vis: Visibility solutions for redundant baseline groups after
    relative calibration
    :param rel_vis: ndarray
    :param distribution: Distribution to fit likelihood {'gaussian', 'cauchy'}
    :type distribution: str
    :param ref_ant_idx: Index of reference antenna in ordered list of antennas to
    constrain overall phase. Default is 16 (corresponding to antenna 55 in H1C_IDR2
    dataset).
    :type ref_ant_idx: int
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
        xgamps = np.ones(no_ants) # gain amplitudes
        xgphases = np.zeros(no_ants) # gain phases
        xgains = np.ravel(np.vstack((xgamps, xgphases)), order='F')
        xdegparams = np.zeros(4) # overall amplitude, overall phase,
        # and phase gradients in x and y
        initp= numpy.hstack([xgains, *xdegparams])

    lb = numpy.repeat(-np.inf, initp.size)
    ub = numpy.repeat(np.inf, initp.size)
    lb[:no_ants*2:2] = 0 # lower bound for gain amplitudes
    lb[-4] = 0 # lower bound for overall amplitude
    bounds = Bounds(lb, ub)

    # constraints for optimization
    constraints = Opt_Constraints(no_ants, ref_ant_idx, ant_pos)
    cons = [{'type': 'eq', 'fun': constraints.avg_amp},
            {'type': 'eq', 'fun': constraints.avg_phase},
            {'type': 'eq', 'fun': constraints.ref_phase},
            # {'type': 'eq', 'fun': constraints.ref_amp}, # not needed (?)
            {'type': 'eq', 'fun': constraints.phase_grad_x},
            {'type': 'eq', 'fun': constraints.phase_grad_y}
            ]

    ff = jit(functools.partial(optimal_logLkl, credg, distribution, \
                               ant_sep, obsvis, rel_vis, no_ants))
    res = minimize(ff, initp, jac=jacrev(ff), hess=jacfwd(jacrev(ff)), \
                   constraints=cons, bounds=bounds, method='trust-constr', \
                   options={'maxiter':max_nit})
    print(res['message'])
    return res


def deg_logLkl(distribution, ant_sep, rel_vis1, rel_vis2, params):
    """Degenerate likelihood calculator

    Max-likelihood estimate to solve for the degenerate parameters that transform
    between the visibility solutions of two different datasets after relative
    calibration

    :param distribution: Distribution to fit likelihood {'gaussian', 'cauchy'}
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
    log_likelihood = LLFN[distribution](delta)
    return log_likelihood


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
    :param distribution: Distribution to fit likelihood {'gaussian', 'cauchy'}
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

    ff = jit(functools.partial(deg_logLkl, distribution, ant_sep, \
                               rel_vis1, rel_vis2))
    res = minimize(ff, initp, jac=jacrev(ff), options={'maxiter':max_nit})
    print(res['message'])
    return res
