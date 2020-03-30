import functools
import itertools

import numpy
from scipy.optimize import minimize
import scipy.stats as stats
import pandas as pd

from jax.config import config
config.update("jax_enable_x64", True)
import jax
from jax import jit, jacrev

# NB. Where "numpy" is used below it has to be real numpy. "np" can be
# either jax or real numpy
np=jax.np

from matplotlib import pyplot as plt
import seaborn as sns

sns.set()
sns.set_style("whitegrid")

from hera_cal.io import HERAData, HERACal
from hera_cal.redcal import get_reds


def fltBad(bls, badl,
           minbl=2):
    """Filter bad baselines

    :param badl: List of bad antennas
    :param minbl: Minimum number of observed baselines in a group
    """
    r1 = map(functools.partial(filter, lambda x: not (x[0] in badl or x[1] \
                               in badl)), bls)
    r2 = list(map(list, r1))
    return list(filter(lambda x: len(x) >= minbl, r2))


def groupBls(bll):
    """Group redundant baselines to indices of unique baselines"""
    return np.array([(g, i, j) for (g, bl) in enumerate(bll) for (i, j, p) in bl])


def condenseMap(a):
    """Return a mapping of indices to a dense index set"""
    return dict(map(reversed, enumerate(numpy.unique(a))))


def relabelAnts(bl_groups):
    """Relabel antennas with consecutive numbering

    :param bl_groups: Grouping of reundant baselines, as returned by the
                      groupBls function
    """
    ci = condenseMap(bl_groups[:, 1:3])
    bl_groups = numpy.copy(bl_groups)
    for i in range(len(bl_groups)):
        bl_groups[i, 1] = ci[bl_groups[i, 1]]
        bl_groups[i, 2] = ci[bl_groups[i, 2]]
    return bl_groups


def redblMap(bl_grouping):
    """Return unique baseline types"""
    bl_ids = numpy.unique(bl_grouping[:, 0], return_index=True)
    return numpy.array(bl_grouping[bl_ids[1], :])


def red_ant_pos(redg, ant_pos):
    """Return positions of the antennas that define a redundant group

    Returns the antenna positions of the baseline that is representative of
    the redundant baseline group it is in (taken to be the baseline with
    the lowest antenna numbers in the group)
    e.g baseline_id 2 has baselines of type (12, 13), therefore we take the positions
    of antennas (12, 13)

    :param redg: Redundant groups
    :param ant_pos: Antenna positions from HERAData container
    """
    redbl_types = redblMap(redg)
    redant_positions = np.array([np.array([ant_pos[i[1]], ant_pos[i[2]]]) \
                       for i in redbl_types])
    return redant_positions


def red_ant_sep(redg, ant_pos):
    """Return seperation of the antennas that define a redundant group

    Seperation defined to be antenna 2 minus antenna 1 in antenna pair

    :param redg: Redundant groups
    :param ant_pos: Antenna positions from HERAData container
    """
    redant_positions = red_ant_pos(redg, ant_pos)
    redant_seperation = redant_positions[:, 1, :] - redant_positions[:, 0, :]
    return redant_seperation


def reformatCArray(arr):
    """Reformat 1D complex array into 1D real array

    The 1D complex array with entries [z_1, ..., z_i] is reformatted such
    that the new array has entries [Re(z_1), Im(z_1), ..., Re(z_i), Im(z_i)],
    which can be easily passed into the likelihood calculators.
    """
    assert arr.ndim == 1
    assert arr.dtype  == numpy.complex
    new_arr = np.vstack((arr.real, arr.imag))
    return new_arr.transpose().flatten()


@jit
def gVis(vis, redg, gains):
    """Apply gains to visibilities

    :param vis: Visibilities
    :param redg: Redundant groups
    """
    return vis[redg[:, 0]]*gains[redg[:, 1]]*np.conj(gains[redg[:, 2]])


def relative_logLkl(redg, distribution, obsvis, params):
    """Redundant relative likelihood calculator

    We set the noise for each visibility to be 1. Note parameter order
    is such that function can be usefully partially applied.

    :param redg: Redundant groups
    :param distribution: Distribution to fit likelihood
    :param obsvis: Observed sky visibilities
    """
    NRedVis = redg[:, 0].max().item() + 1
    vis_comps, gains_comps = np.split(params, [NRedVis*2, ])
    vis_comps = vis_comps.reshape((-1, 2))
    gains_comps = gains_comps.reshape((-1, 2))

    vis = vis_comps[:, 0]+1j*vis_comps[:, 1]
    gains = gains_comps[:, 0]+1j*gains_comps[:, 1]

    delta = obsvis - gVis(vis, redg, gains)

    if distribution == 'gaussian':
        log_likelihood = np.square(np.abs(delta)).sum() # omit factor of 0.5
    elif distribution == 'cauchy':
        log_likelihood = np.log(1 + np.square(np.abs(delta))).sum()
    else:
        raise ValueError('Specify correct type of distribution for MLE estimation')

    return log_likelihood


def optimal_logLkl(redg, distribution, obsvis, ant_sep, red_vis_comps, params):

    """Optimal likelihood calculator

    We solve for the degeneracies in redundant calibration. This must be done
    after relative redundant calibration. We also set the noise for each visibility
    to be 1.

    :param redg: Redundant groups
    :param distribution: Distribution to fit likelihood
    :param obsvis: Observed sky visibilities
    :param ant_sep: Antenna seperation for baseline types
    :param red_vis: Relative MLE visibilities for redundant baseline groups
    :param params: Parameters to be estimated - normalized gains, overall amplitude
                   phase gradients in x and y
     """
    NAnts = redg[:, 1:].max().item() + 1
    rel_gains_comps, deg_params = np.split(params, [2*NAnts,])
    amp, overall_phase, phase_grad_x, phase_grad_y = deg_params
    red_vis_comps = red_vis_comps.reshape((-1, 2))
    rel_gains_comps = rel_gains_comps.reshape((-1, 2))

    red_vis = red_vis_comps[:, 0] + 1j*red_vis_comps[:, 1]
    rel_gains = rel_gains_comps[:, 0] + 1j*rel_gains_comps[:, 1]
    x_sep = ant_sep[:, 0]
    y_sep = ant_sep[:, 1]

    w_alpha  = np.square(amp) * np.exp(1j * (phase_grad_x * x_sep + phase_grad_y \
               * y_sep)) * (red_vis)

    delta = obsvis - gVis(w_alpha, redg, rel_gains)

    if distribution == 'gaussian':
         log_likelihood = np.square(np.abs(delta)).sum()
    elif distribution == 'cauchy':
        log_likelihood = np.log(1 + np.square(np.abs(delta))).sum()
    else:
        raise ValueError('Specify correct type of distribution for MLE estimation')

    return log_likelihood


def pvis(v):
    plt.plot(v.real, label="real")
    plt.plot(v.imag, label="imag")
    plt.legend()
    plt.savefig("plots/vis.png")
    plt.clf()
