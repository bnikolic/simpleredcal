"""Robust redundant calibration"""


import functools

import numpy
from matplotlib import pyplot as plt
from scipy import stats as stats
from scipy.optimize import Bounds, minimize

import hera_cal
from hera_cal.io import HERACal, HERAData
from hera_cal.redcal import get_reds

from jax.config import config
config.update('jax_enable_x64', True)
import jax
from jax import jit, jacrev

# n.b. where 'numpy' is used below it has to be real numpy. 'np' can be
# either jax or real numpy
np=jax.np


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
        print('{} out of {} data points flagged for dataset {}'.format(no_flags, \
              data_size, zen_path))

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
    redbl_types = redblMap(redg)
    redant_positions = np.array([np.array([ant_pos[i[1]], ant_pos[i[2]]]) \
                       for i in redbl_types])
    return redant_positions


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
    redant_positions = red_ant_pos(redg, ant_pos)
    redant_seperation = redant_positions[:, 0, :] - redant_positions[:, 1, :]
    return redant_seperation


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
    w_alpha = np.square(amp) * np.exp(1j * (phase_grad_x * x_sep + phase_grad_y \
               * y_sep)) * rel_vis
    return w_alpha


def relative_logLkl(credg, distribution, obsvis, params):
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
    :param params: Parameters to constrain: redundant visibilities and gains
    (Re & Im components interweaved for both)
    :type params: ndarray

    :return: Negative log-likelihood of MLE computation
    :rtype: float
    """
    NRedVis = credg[:, 0].max().item() + 1
    vis_comps, gains_comps = np.split(params, [NRedVis*2, ])
    vis = makeCArray(vis_comps)
    gains = makeCArray(gains_comps)

    delta = obsvis - gVis(vis, credg, gains)
    log_likelihood = LLFN[distribution](delta)
    return log_likelihood


def optimal_logLkl(credg, distribution, ant_sep, obsvis, rel_vis, params):
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
    NAnts = credg[:, 1:].max().item() + 1
    rel_gains_comps, deg_params = np.split(params, [2*NAnts,])
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

    :param ants: Antenna numbers dealt with in visibility dataset
    :type ants: ndarray
    :param ref_ant: Antenna number of reference antenna to constrain overall phase
    :type ref_ant: int
    :param ant_pos: Dictionary of antenna position coordinates for the antennas
    in ants
    :type ant_pos: dict
    :param params: Parameters to feed into optimal absolute calibration
    :type params: ndarray
    """
    def __init__(self, ants, ref_ant, ant_pos, redg):
        self.ants = ants
        self.ref_ant = ref_ant
        self.ant_pos = ant_pos
        self.redg = redg
        self.ref_ant_idx = condenseMap(self.ants)[self.ref_ant]
        self.cmap = condenseMap(self.redg[:, 1:3])
        self.x_pos = np.asarray([self.ant_pos[ant_no][0] for ant_no in self.cmap.keys()])
        self.y_pos = np.asarray([self.ant_pos[ant_no][1] for ant_no in self.cmap.keys()])

    def get_rel_gains(self, params):
        """Returns the complex relative gain parameters from the flattened array
        of parameters

        :return: Relative gain parameters in complex array format
        :rtype: ndarray
        """
        rel_gains_comps = params[:self.ants.size*2]
        return makeEArray(rel_gains_comps)

    def avg_amp(self, params):
        """Constraint that mean of gain amplitudes must be equal to 1

        :return: Residual between mean gain amplitudes and 1
        :rtype: float
        """
        amps = params[:self.ants.size*2:2]
        return np.mean(amps) - 1

    def avg_phase(self, params):
        """Constraint that mean of gain phases must be equal to 0

        :return: Residual between circular mean of gain phases and 0
        :rtype: float
        """
        phases = params[1:self.ants.size*2:2]
        return np.mean(phases)

    def ref_phase(self, params):
        """Set phase of reference antenna gain to 0 to set overall phase

        :return: Residual between referance antenna phase and 0
        :rtype: float
        """
        phases = params[1:self.ants.size*2:2]
        return phases[self.ref_ant_idx]

    def ref_amp(self, params):
        """Set amplitude of reference antenna gain to 1 to set overall amplitude

        :return: Residual between referance antenna amplitude and 1
        :rtype: float
        """
        amps = params[:self.ants.size*2:2]
        return amps[self.ref_ant_idx] - 1

    def phase_grad_x(self, params):
        """Constraint that phase gradient in x is 0

        :return: Residual between phase gradient in x and 0
        :rtype: float
        """
        phases = params[1:self.ants.size*2:2]
        return np.sum(phases*self.x_pos)

    def phase_grad_y(self, params):
        """Constraint that phase gradient in y is 0

        :return: Residual between phase gradient in y and 0
        :rtype: float
        """
        phases = params[1:self.ants.size*2:2]
        return np.sum(phases*self.y_pos)


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
    w_alpha = degVis(ant_sep, rel_vis1, *params[[0, 2, 3]])
    delta = rel_vis2 - w_alpha
    log_likelihood = LLFN[distribution](delta)
    return log_likelihood


def doRelCal(redg, obsvis, distribution='cauchy', initp=None):
    """Do relative step of redundant calibration

    Initial parameter guesses are 1+1j for both the gains and the true
    sky visibilities.

    :param redg: Grouped baselines, as returned by groupBls
    :type redg: ndarray
    :param obsvis: Observed sky visibilities for a given frequency and given time,
    reformatted to have format consistent with redg
    :type obsvis: ndarray
    :param distribution: Distribution to fit likelihood {'gaussian', 'cauchy'}
    :type distribution: str
    :param initp: Initial parameter guesses for true visibilities and gains
    :type initp: ndarray, None

    :return: Optimization result for the solved antenna gains and true sky
    visibilities
    :rtype: Scipy optimization result object
    """
    if initp is None:
        # setup initial parameters
        ants = numpy.unique(redg[:, 1:])
        no_unq_bls = numpy.unique(redg[:, 0]).size
        xvis = numpy.ones(no_unq_bls*2) # complex vis
        xgains = numpy.ones(ants.size*2) # complex gains (Re & Im components)
        initp = numpy.hstack([xvis, xgains])

    ff = jit(functools.partial(relative_logLkl, relabelAnts(redg), \
                               distribution, obsvis))
    res = minimize(ff, initp, jac=jacrev(ff))
    print(res['message'])
    return res


def doOptCal(redg, obsvis, ant_pos, rel_vis, distribution='cauchy', ref_ant=12, \
             initp=None):
    """Do optimal absolute step of redundant calibration

    Initial degenerate parameter guesses are 1 for the overall amplitude, and 0
    for the overall phase and the phase gradients in x and y.

    :param redg: Grouped baselines, as returned by groupBls
    :type redg: ndarray
    :param obsvis: Observed sky visibilities for a given frequency and given time,
    reformatted to have format consistent with redg
    :type obsvis: ndarray
    :param ant_pos: Dictionary of antenna position coordinates for the antennas
    in ants
    :type ant_pos: dict
    :param rel_vis: Visibility solutions for redundant baseline groups after
    relative calibration
    :param rel_vis: ndarray
    :param distribution: Distribution to fit likelihood {'gaussian', 'cauchy'}
    :type distribution: str
    :param ref_ant: Antenna number of reference antenna to constrain overall phase
    :type ref_ant: int
    :param initp: Initial parameter guesses for gains and degenerate parameters
    :type initp: ndarray, None

    :return: Optimization result for the optimally absolutely calibrated
    redundant gains and degenerate parameters
    :rtype: Scipy optimization result object
    """
    ants = numpy.unique(redg[:, 1:])
    if initp is None:
        # setup initial parameters
        xgamps = np.ones(ants.size) # gain amplitudes
        xgphases = np.zeros(ants.size) # gain phases
        xgains = np.ravel(np.vstack((xgamps, xgphases)), order='F')
        xdegparams = np.zeros(4) # overall amplitude, overall phase,
        # and phase gradients in x and y
        initp= np.hstack([xgains, *xdegparams])

    lb = numpy.repeat(-np.inf, initp.size)
    ub = numpy.repeat(np.inf, initp.size)
    lb[:ants.size:2] = 0 # lower bound for gain amplitudes
    lb[-4] = 0 # lower bound for overall amplitude
    bounds = Bounds(lb, ub)

    # constraints for optimization
    constraints = Opt_Constraints(ants, ref_ant, ant_pos, redg)
    cons = [{'type': 'eq', 'fun': constraints.avg_amp},
            {'type': 'eq', 'fun': constraints.avg_phase},
            {'type': 'eq', 'fun': constraints.ref_phase},
            # {'type': 'eq', 'fun': constraints.ref_amp}, # not needed (?)
            {'type': 'eq', 'fun': constraints.phase_grad_x},
            {'type': 'eq', 'fun': constraints.phase_grad_y}
            ]

    ant_sep = red_ant_sep(redg, ant_pos)
    ff = jit(functools.partial(optimal_logLkl, relabelAnts(redg), distribution, \
                               ant_sep, obsvis, rel_vis))
    res = minimize(ff, initp, constraints=cons, jac=jacrev(ff), \
                   method='trust-constr', bounds=bounds)
    print(res['message'])
    return res


def doDegVisVis(redg, ant_pos, rel_vis1, rel_vis2, distribution='cauchy', \
                initp=None):
    """
    Fit degenerate redundant calibration parameters so that rel_vis1 is as
    close to as possible to rel_vis1

    :param redg: Grouped baselines, as returned by groupBls. These should be the
    same for datasets 1 and 2.
    :type redg: ndarray
    :param ant_pos: Dictionary of antenna position coordinates for the antennas
    in ants
    :type ant_pos: dict
    :param rel_vis1: Visibility solutions for observation set 1 for redundant
    baseline groups after relative calibration
    :param rel_vis1: ndarray
    :param rel_vis2: Visibility solutions for observation set 2 for redundant
    baseline groups after relative calibration
    :param rel_vis2: ndarray
    :param initp: Initial parameter guesses for degenerate parameters
    :type initp: ndarray, None

    :return: Optimization result for the solved degenerate parameters that
    translat between the two datasets
    :rtype: Scipy optimization result object
    """
    if initp is None:
        # setup initial parameters: overall amplitude, overall phase, and x and y
        # phase gradients
        initp = np.asarray([1, 0, 0, 0])

    ant_sep = red_ant_sep(redg, ant_pos)
    ff = jit(functools.partial(deg_logLkl, distribution, ant_sep, \
                               rel_vis1, rel_vis2))
    res = minimize(ff, initp, jac=jacrev(ff))
    print(res['message'])
    return res
