import functools
import itertools

import numpy
import scipy
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
    r1 = map(functools.partial(filter, lambda x: not (x[0] in badl or x[1] in badl)), bls)
    r2 = list(map(list, r1))
    return list(filter(lambda x: len(x) >= minbl, r2))


def groupBls(bll):
    """Group reundant baselines to indices of unique baselines"""
    return np.array([(g, i, j) for (g, bl) in enumerate(bll) for (i, j, p) in bl])


def condenseMap(a):
    """Return a mapping of indicies to a dense index set"""
    return dict(map(reversed, enumerate(numpy.unique(a))))


def relabelAnts(a):
    """Relabel antennas with consecutive numbering"""
    ci = condenseMap(a[:, 1:3])
    a = numpy.copy(a)
    for i in range(len(a)):
        a[i, 1] = ci[a[i, 1]]
        a[i, 2] = ci[a[i, 2]]
    return a

@jit
def gVis(vis, redg, gains):
    """Apply gains to visibilities

    :param vis: true sky visibilities
    """
    return vis[redg[:, 0]]*gains[redg[:, 1]]*np.conj(gains[redg[:, 2]])

def vgLkl(redg, dist, p, obsvis):
    """Simple likelihood calculator

    We set the noise for each visibility to be 1. Note parameter order
    is such that function can be usefully partially applied

    :param redg: Reundant groups
    """
    NVis = redg[:, 0].max().item()+1
    vis, gains = np.split(p, [NVis*2, ])
    vis = vis.reshape((-1, 2))
    gains = gains.reshape((-1, 2))
    delta = obsvis-gVis(vis[:, 0]+1j*vis[:, 1], redg, gains[:, 0]+1j*gains[:, 1])
    if dist == 'gaussian':
        likelihood = (0.5*np.square(np.abs(delta))).sum()
    elif dist == 'cauchy':
        likelihood = (np.log(1 + np.square(np.abs(delta)))).sum()
    else:
        raise ValueError('Specify correct type of distribution for MLE estimation')
    return likelihood


def pvis(v):
    pyplot.plot(v.real, label="real")
    pyplot.plot(v.imag, label="imag")
    pyplot.legend()
    pyplot.savefig("plots/vis.png")
    pyplot.clf()

