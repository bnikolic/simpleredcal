"""Robust redundant calibration"""


import functools

import numpy
from matplotlib import pyplot as plt
from scipy import stats as stats
from scipy.optimize import minimize

from hera_cal.io import HERAData
from hera_cal.redcal import get_reds

from jax.config import config
config.update('jax_enable_x64', True)
import jax
from jax import jit, jacrev

# NB. Where "numpy" is used below it has to be real numpy. "np" can be
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


def group_data(zen_path, pol, freq_chans, bad_ants):
    """Returns redundant baseline grouping and reformatted dataset

    :param zen_path: Path of uvh5 dataset
    :type zen_path: str
    :param pol: Polarization of data
    :type pol: str
    :param freq_chans: Frequency channel(s) {0, 1023} (None to choose all)
    :type freq_chans: array-like, or None
    :param bad_ants: Known bad antennas to flag
    :type bad_ants: array-like

    :return hd: HERAData class
    :rtype hd: HERAData class
    :return redg: Grouped baselines, as returned by groupBls
    :rtype redg: ndarray
    :return cdata: Grouped visibilities with format consistent with redg and
    dimensions (freq chans, time integrations, baselines) - the 0th dimension
    only has 1 element if only 1 frequency channel is specified
    :rtype cdata: ndarray
    """
    hd = HERAData(zen_path)
    reds = get_reds(hd.antpos, pols=[pol])
    # if isinstance(freq_chans, int):
    #     freq_chans = [freq_chans]
    data, flags, nsamples = hd.read(freq_chans=freq_chans)
    flt_bls = fltBad(reds, bad_ants)
    redg = groupBls(flt_bls) # Baseline grouping
    no_tints, no_chans = data[list(data.keys())[0]].shape
    if freq_chans is None:
        freq_chans = numpy.arange(no_chans)

    # Collect data together
    cdata = numpy.empty((len(freq_chans), no_tints, redg.shape[0]), \
                        dtype=complex)
    for idx in range(len(freq_chans)):
        cdata[idx, ...] = numpy.hstack([data[(*bl_row[1:], pol)][:, idx, \
                                        numpy.newaxis] for bl_row in redg])
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
    redundant sets
    :rtype: ndarray
    """
    redbl_types = redblMap(redg)
    redant_positions = np.array([np.array([ant_pos[i[1]], ant_pos[i[2]]]) \
                       for i in redbl_types])
    return redant_positions


def red_ant_sep(redg, ant_pos):
    """Return seperation of the antennas that define a redundant group

    Seperation defined to be antenna 2 minus antenna 1 in antenna pair

    :param redg: Grouped baselines, as returned by groupBls
    :type redg: ndarray
    :param ant_pos: Antenna positions from HERAData container
    :type ant_pos: dict

    :return: Array of seperations of the baselines that define the redundant sets
    :rtype: ndarray
    """
    redant_positions = red_ant_pos(redg, ant_pos)
    redant_seperation = redant_positions[:, 1, :] - redant_positions[:, 0, :]
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
    """Reformat 1D real array into 1D complex array

    The 1D real array with elements [Re(z_1), Im(z_1), ..., Re(z_i), Im(z_i)]
    is reformatted such that the new array has elements [z_1, ..., z_i].

    :param arr: Real array where the complex elements of arr have been decomposed
    into adjacent real elements
    :type arr: ndarray

    :return: Complex array
    :rtype: ndarray
    """
    assert arr.ndim == 1
    assert arr.dtype  == numpy.float
    arr = arr.reshape((-1, 2))
    return arr[:, 0] + 1j*arr[:, 1]


@jit
def gVis(vis, redg, gains):
    """Apply gains to visibilities

    :param vis: visibilities
    :type vis: ndarray
    :param redg: Grouped baselines, as returned by groupBls
    :type redg: ndarray
    :param gains: Antenna gains
    :type gains: ndarray

    :return: Modified visibilities by applying antenna gains
    :rtype: ndarray
    """
    return vis[redg[:, 0]]*gains[redg[:, 1]]*np.conj(gains[redg[:, 2]])


# Could also put these in a module and then use getattr
LLFN = { 'cauchy' : lambda delta: np.log(1 + np.square(np.abs(delta))).sum(),
         'gaussian' : lambda delta: np.square(np.abs(delta)).sum() }


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


def relative_logLkl(redg, distribution, obsvis, params):
    """Redundant relative likelihood calculator

    We impose that the true sky visibilities from redundant baseline sets are
    equal, and use this prior to calibrate the visibilities (up to some degenerate
    parameters). We set the noise for each visibility to be 1.

    Note: parameter order is such that the function can be usefully partially applied.

    :param redg: Grouped baselines, as returned by groupBls
    :type redg: ndarray
    :param distribution: Distribution to fit likelihood {'gaussian', 'cauchy'}
    :type distribution: str
    :param obsvis: Observed sky visibilities for a given frequency and given time,
    reformatted to have format consistent with redg
    :type obsvis: ndarray
    :param params: Parameters to constrain: redundant visibilities and gains
    :type params: ndarray

    :return: Negative log-likelihood of MLE computation
    :rtype: ndarray (1 element)
    """
    NRedVis = redg[:, 0].max().item() + 1
    vis_comps, gains_comps = np.split(params, [NRedVis*2, ])
    vis = makeCArray(vis_comps)
    gains = makeCArray(gains_comps)

    delta = obsvis - gVis(vis, redg, gains)
    log_likelihood = LLFN[distribution](delta)
    return log_likelihood


def optimal_logLkl(redg, distribution, ant_sep, obsvis, rel_vis, params):
    """Optimal likelihood calculator

    We solve for the degeneracies in redundant calibration. This must be done
    after relative redundant calibration. We also set the noise for each visibility
    to be 1.

    :param redg: Grouped baselines, as returned by groupBls
    :type redg: ndarray
    :param distribution: Distribution to fit likelihood {'gaussian', 'cauchy'}
    :type distribution: str
    :param obsvis: Observed sky visibilities for a given frequency and given time,
    reformatted to have format consistent with redg
    :type obsvis: ndarray
    :param ant_sep: Antenna seperation for baseline types
    :type ant_sep: ndarray
    :param rel_vis: Visibility solutions for redundant baseline groups after
    relative calibration
    :param rel_vis: ndarray
    :param params: Parameters to constrain: normalized gains, overall amplitude,
    overall phase and phase gradients in x and y
    :type params: ndarray

    :return: Negative log-likelihood of MLE computation
    :rtype: ndarray (1 element)
     """
    NAnts = redg[:, 1:].max().item() + 1
    rel_gains_comps, deg_params = np.split(params, [2*NAnts,])
    rel_gains = makeCArray(rel_gains_comps)

    w_alpha = degVis(ant_sep, rel_vis, *deg_params[[0, 2, 3]])
    delta = obsvis - gVis(w_alpha, redg, rel_gains)
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
    """

    def __init__(self, ants, ref_ant, ant_pos):
        self.ants = ants
        self.ref_ant = ref_ant
        self.ant_pos = ant_pos

    def get_rel_gains(self, params):
        """Returns the complex relative gain parameters from the flattened array
        of parameters"""
        rel_gains_comps = params[:self.ants.size*2]
        return makeCArray(rel_gains_comps)

    def avg_amp(self, params):
        """Constraint that average of gain amplitudes must be equal to 1"""
        rel_gains = self.get_rel_gains(params)
        return np.average(np.abs(rel_gains)) - 1

    def avg_phase(self, params):
        """Constraint that average of gain phases must be equal to 0"""
        rel_gains = self.get_rel_gains(params)
        return stats.circmean(np.angle(rel_gains))

    def ref_phase(self, params):
        """Set argument of reference antenna to zero to set overall phase"""
        rel_gains = self.get_rel_gains(params)
        ref_ant_idx = condenseMap(self.ants)[self.ref_ant]
        return np.angle(rel_gains[ref_ant_idx])

    def phase_grad(self, params):
        """Constraint that phase gradient is zero

        TODO: does the phase gradient need to be made zero across all gains?
        """
        deg_params = params[-4:] # set degenerate parameters at the end
        _, overall_phase, phase_grad_x, phase_grad_y = deg_params
        x_ant_ref_pos, y_ant_ref_pos = self.ant_pos[self.ref_ant][:2]
        phase_gradient = overall_phase + (x_ant_ref_pos * phase_grad_x) + \
                         (y_ant_ref_pos * phase_grad_y)
        return phase_gradient


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
    :rtype: ndarray (1 element)
    """
    w_alpha = degVis(ant_sep, rel_vis1, *params[[0, 2, 3]])
    delta = rel_vis2 - w_alpha
    log_likelihood = LLFN[distribution](delta)
    return log_likelihood


def doRelCal(redg, obsvis, distribution='cauchy'):
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

    :return: Optimization result for the solved antenna gains and true sky
    visibilities
    :rtype: Scipy optimization result object
    """
    #Setup initial parameters
    ants = numpy.unique(redg[:, 1:])
    no_unq_bls = numpy.unique(redg[:, 0]).size
    xvis = numpy.ones(no_unq_bls*2) # Complex vis
    xgains = numpy.ones(ants.size*2) # Complex gains
    initp = numpy.hstack([xvis, xgains])

    ff = jit(functools.partial(relative_logLkl, relabelAnts(redg), \
                               distribution, obsvis))
    res = minimize(ff, initp, jac=jacrev(ff))
    print(res['message'])
    return res


def doOptCal(redg, obsvis, ant_pos, rel_vis, distribution='cauchy', ref_ant=12):
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

    :return: Optimization result for the optimally absolutely calibrated
    redundant gains and degenerate parameters
    :rtype: Scipy optimization result object
    """
    #Setup initial parameters
    ants = numpy.unique(redg[:,1:])
    xgains = numpy.ones(ants.size*2) # Complex gains
    xdegparams = np.asarray([1, 0, 0, 0]) # Overall amplitude, overall phase, and phase
    # gradients in x and y
    initp= numpy.hstack([xgains, *xdegparams])

    # Constraints for optimization
    constraints = Opt_Constraints(ants, ref_ant, ant_pos)
    cons = [{'type': 'eq', 'fun': constraints.avg_amp},
            {'type': 'eq', 'fun': constraints.avg_phase},
            {'type': 'eq', 'fun': constraints.ref_phase},
            {'type': 'eq', 'fun': constraints.phase_grad}]

    ant_sep = red_ant_sep(redg, ant_pos)
    ff = jit(functools.partial(optimal_logLkl, relabelAnts(redg), distribution, \
                               ant_sep, obsvis, rel_vis))
    res = minimize(ff, initp, constraints=cons, jac=jacrev(ff), \
                   method='trust-constr')
    print(res['message'])
    return res


def doDegVisVis(redg, ant_pos, rel_vis1, rel_vis2, distribution='cauchy'):
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

    :return: Optimization result for the solved degenerate parameters that
    translat between the two datasets
    :rtype: Scipy optimization result object
    """
    #Setup initial parameters: overall amplitude and phase, and x & y phase gradients
    initp = np.asarray([1, 0, 0, 0])
    ant_sep = red_ant_sep(redg, ant_pos)
    ff = jit(functools.partial(deg_logLkl, distribution, ant_sep, \
                               rel_vis1, rel_vis2))
    res = minimize(ff, initp, jac=jacrev(ff))
    print(res['message'])
    return res
