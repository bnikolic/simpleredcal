from hera_cal.io import HERAData, HERACal
from hera_cal.redcal import get_reds
import functools
import itertools

import numpy as np
import scipy

import pandas as pd
from scipy.optimize import minimize
import scipy.stats as stats
import seaborn as sns

sns.set()
sns.set_style("whitegrid")


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
    return dict(map(reversed, enumerate(np.unique(a))))


def relabelAnts(a):
    """Relabel antennas with consequitive numbering"""
    ci = condenseMap(a[:, 1:3])
    a = a.copy()
    for i in range(len(a)):
        a[i, 1] = ci[a[i, 1]]
        a[i, 2] = ci[a[i, 2]]
    return a


def gVis(vis, redg, gains):
    """Apply gains to visibilities

    :param vis: true sky visibilities
    """
    redg = relabelAnts(redg)
    return vis[redg[:, 0]]*gains[redg[:, 1]]*np.conj(gains[redg[:, 2]])


def vgLkl(p, redg, obsvis, dist='gaussian'):
    """Simple likelihood calculator

    We set the noise for each visibility to be 1.

    :param redg: Reundant groups
    """
    NVis = redg[:, 0].max()+1
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
