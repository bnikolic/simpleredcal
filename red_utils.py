"""Utility functions for robust redundant calibration"""


import os
import pickle

import numpy
from matplotlib import pyplot as plt
from scipy.optimize import minimize_scalar

import pyuvdata
from pyuvdata import utils as uvutils

from red_likelihood import makeCArray


def find_zen_file(JD_time):
    """Returns path of selected JD_time

    :param JD_time: Fractional Julian date
    :type JD_time: str

    :return: File path of visibility dataset
    :rtype: str
    """
    mdm_dir = '/Users/matyasmolnar/Downloads/HERA_Data/robust_cal'
    nrao_dir = '/lustre/aoc/projects/hera/H1C_IDR2'
    zen_file = 'zen.{}.HH.uvh5'.format(JD_time)
    jd_day = zen_file.split('.')[1]
    if os.path.exists(mdm_dir):
        zen_path = os.path.join(mdm_dir, zen_file)
    elif os.path.exists(nrao_dir):
        zen_path = os.path.join(nrao_dir, jd_day, zen_file)
    else:
        zen_path = './{}'.format(zen_file)

    if not os.path.exists(zen_path):
        raise ValueError('Dataset {} not found'.format(zen_file))
    return zen_path


def fn_format(fn, format):
    """Format file name to have correct file extension

    :param figname: File name
    :type figname: str
    :param format: File format
    :type format: str

    :return: Formatted figure name
    :rtype: str
    """
    ext = '.{}'.format(format)
    if not fn.endswith(ext):
        fn = fn + ext
    return fn


def get_bad_ants(zen_path):
    """Find the corresponding bad antennas for the given IDR2 visibility
     dataset

    :param zen_path: File path of visibility dataset
    :type zen_path: str

    :return: Bad antennas
    :rtype: ndarray
     """
    jd_day = int(os.path.basename(zen_path).split('.')[1])
    with open('bad_ants_idr2.pkl', 'rb') as f:
        bad_ants_dict = pickle.load(f)
    return bad_ants_dict[jd_day]


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


def split_rel_results(res, no_unq_bls):
    """Split results from minimization into visibility and gains arrays

    :param res: Optimization result for the solved antenna gains and true sky
    visibilities
    :type res: Scipy optimization result object
    :param no_unq_bls: Number of unique baselines / number of redundant visibilities
    :type no_unq_bls: int

    """
    vis_params, gains_params = numpy.split(res['x'], [no_unq_bls*2,])
    res_vis = makeCArray(vis_params)
    res_gains = makeCArray(gains_params)
    return res_vis, res_gains


def norm_residuals(x_meas, x_pred):
    """Evaluates the residual between the measured and predicted quantities,
    normalized by the absolute value of their product

    :param x_meas: Measurement
    :type x_meas: ndarray
    :param x_pred: Prediction
    :type x_pred: ndarray

    :return: Normalized residual
    :rtype: ndarray
    """
    return (x_meas - x_pred) / numpy.sqrt(numpy.abs(x_meas)*numpy.abs(x_pred))


def abs_residuals(residuals):
    """Return median absolute residuals for both real and imag parts

    :param residuals: Residuals
    :type residuals: ndarray

    :return: Median absolute residuals, separately for Re and Im
    :rtype: list
    """
    return [numpy.median(numpy.absolute(getattr(residuals, i))) \
            for i in ('real', 'imag')]

def plot_red_vis(cdata, redg, vis_type='amp', figsize=(13, 4)):
    """Pot visibility amplitudes or phases, grouped by redundant type

    :param cdata: Grouped visibilities with format consistent with redg
    :type cdata: ndarray
    :param redg: Grouped baselines, as returned by groupBls
    :type redg: ndarray
    :param vis_type: Plot either visibility amplitude or phase {'amp', 'phase'}
    :type vis_type: str
    """
    vis_calc = {'amp':numpy.abs, 'phase': numpy.angle}
    bl_id_seperations = numpy.unique(redg[:, 0], return_index=True)[1][1:]
    fig, ax = plt.subplots(figsize=figsize)
    ax.matshow(vis_calc[vis_type](cdata), aspect='auto')
    for bl_id_seperation in bl_id_seperations:
        plt.axvline(x=bl_id_seperation, color='white', linestyle='-.', linewidth=1)
    ax.grid(False)
    ax.set_xlabel('Baseline ID')
    ax.set_ylabel('Time Integration')
    plt.show()


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
