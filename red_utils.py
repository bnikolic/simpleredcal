"""Utility functions for robust redundant calibration"""

import numpy
from matplotlib import pyplot as plt
from scipy.optimize import minimize_scalar

import pyuvdata
from pyuvdata import utils as uvutils


def find_nearest(arr, val):
    """Find nearest value in array and its index

    :param array: Array-like
    :type array: array-like
    :param val: Find nearest value to this value
    :type val: float, int

    :return: Tuple of nearest value to val in array and its index
    :rtype: tuple
    """
    arr = numpy.asarray(arr)
    idx = (numpy.abs(arr - val)).argmin()
    return arr[idx], idx


def jd_to_lst(JD_time, telescope='HERA'):
    """Converts fractional JDs of HERAData object into LSTs

    :param JD: Julian time
    :type JD: float
    :param telescope: Known telescope to pyuvdata
    :type telescope: str

    :return: Local (Apparent) sidereal time
    :rtype: float
    """
    lat_lon_alt = pyuvdata.telescopes.get_telescope(telescope).telescope_location_lat_lon_alt_degrees
    lst = uvutils.get_lst_for_time([JD_time], *lat_lon_alt)
    return lst[0]


def lst_to_jd_time(lst, JD_day, telescope='HERA'):
    """Find the JD time for a JD day that corresponds to a given LST

    :param lst: Local (Apparent) sidereal time
    :type lst: float
    :param JD: Julian day
    :type JD: int
    :param telescope: Known telescope to pyuvdata
    :type telescope: str

    :return: Julian time
    :rtype: float
    """
    lat_lon_alt = pyuvdata.telescopes.get_telescope(telescope).telescope_location_lat_lon_alt_degrees

    def func(x):
        return numpy.abs(uvutils.get_lst_for_time([JD_day+x], *lat_lon_alt)[0] - lst)

    res = minimize_scalar(func, bounds=(0, 1), method='bounded')
    return res['x']


def cplot(carr, figsize=(12,8), split_ax=False, save_plot=False, save_dir='plots',
          **kwargs):
    """Plot real and imaginary parts of complex array on same plot

    :param carr: Complex 1D array
    :type carr: ndarray
    :param figsize: Figure size
    :type figsize: tuple
    :param split_ax: Split real and imag components onto separate axes?
    :type split_ax: bool
    :param save_plot: Save plot?
    :type save_plot: bool
    :param save_dir: Path of directory to save plots
    :type save_dir: str
    """
    if not split_ax:
        plt.figure(figsize=figsize)
        plt.plot(carr.real, label='Real')
        plt.plot(carr.imag, label='Imag')
        for k, v in kwargs.items():
            getattr(plt, k)(v)
        plt.legend()
    else:
        fig, ax = plt.subplots(nrows=2, sharex=True, figsize=figsize)
        ax[0].plot(carr.real)
        ax[1].plot(carr.imag)
        ax[0].set_ylabel('Real')
        ax[1].set_ylabel('Imag')
        plt.xlabel('Baseline')
        if 'title' in kwargs.keys():
            fig.suptitle(kwargs['title'])
    if save_plot:
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        plt.savefig('{}/vis.png'.format(save_dir))
    plt.show()
    plt.clf()
