"""Utility functions for robust redundant calibration"""


import datetime
import os
import pickle
import re

import numpy
import pandas as pd
from scipy.optimize import minimize_scalar

import pyuvdata
from pyuvdata import utils as uvutils

from red_likelihood import makeCArray, makeEArray


def find_zen_file(JD_time):
    """Returns visibility dataset path for the specified JD_time

    :param JD_time: Fractional Julian date
    :type JD_time: float

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


def find_flag_file(JD_time, cal_type):
    """Returns flag dataset path for the specified JD_time

    :param JD_time: Fractional Julian date
    :type JD_time: float
    :param cal_type: Calibration process that produced the flag file {"first",
    "omni", "abs", "flagged_abs", "smooth_abs"}, to name a few
    :type cal_type: str

    :return: File path of visibility dataset
    :rtype: str, None
    """
    mdm_dir = '/Users/matyasmolnar/Downloads/HERA_Data/robust_cal'
    nrao_dir = '/lustre/aoc/projects/hera/H1C_IDR2/IDR2_2'
    flg_file = 'zen.{}.HH.{}.calfits'.format(JD_time, cal_type)
    jd_day = flg_file.split('.')[1]
    if os.path.exists(mdm_dir):
        flg_path = os.path.join(mdm_dir, flg_file)
    elif os.path.exists(nrao_dir):
        flg_path = os.path.join(nrao_dir, jd_day, flg_file)
    else:
        flg_path = './{}'.format(flg_file)

    if not os.path.exists(flg_path):
        print('Flag file {} not found\n'.format(flg_file))
        flg_path = None
    return flg_path


def find_rel_df(JD_time, pol, dist):
    """Returns relative calibration results dataframe path for the specified
    JD_time

    :param JD_time: Fractional Julian date
    :type JD_time: str
    :param pol: Polarization of data
    :type pol: str
    :param dist: Fitting distribution for calibration {"cauchy", "gaussian"}
    :type dist: str

    :return: File path of relative calibration results dataframe
    :rtype: str
    """
    df_path = './rel_df.{}.{}.{}.pkl'.format(JD_time, pol, dist)
    if not os.path.exists(df_path):
        raise ValueError('DataFrame {} not found'.format(df_path))
    return df_path


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


def find_nearest(arr, val, condition=None):
    """Find nearest value in array and its index

    :param array: Array-like
    :type array: array-like
    :param val: Find nearest value to this value
    :type val: float, int
    :param condition: If nearest number must be less or greater than the target
    {"leq", "geq"}
    :type condition: str

    :return: Tuple of nearest value to val in array and its index
    :rtype: tuple
    """
    arr = numpy.asarray(arr)
    if condition is None:
        idx = (numpy.abs(arr - val)).argmin()
    if condition == 'leq':
        idx = numpy.where(numpy.less_equal(arr - val, 0))[0][-1]
    if condition == 'geq':
        idx = numpy.where(numpy.greater_equal(arr - val, 0))[0][0]
    return arr[idx], idx


def jd_to_lst(JD_time, telescope='HERA'):
    """Converts fractional JD of HERAData object into LAST

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
    return JD_day + res['x']


def match_lst(JD_time, JD_day, tint=0):
    """Finds the JD time of the visibility dataset that constains data at a given
    JD_day at the same LAST as the dataset at JD_time.

    e.g. I have dataset at JD_time = 2458098.43869 and I want to find the dataset
    on JD_day = 2458110 that has data for the same LAST as that for JD_time.
    This function retuns the JD time of the dataset that satisfies this
    (2458110.40141 for this example).

    :param JD_time: Julian time of the dataset we want to match in LAST
    :type JD_time: float
    :param JD_day: Julian day of target dataset
    :type JD_day: int
    :param tint: Time integration index to match
    :type tint: int

    :return: Julian time of the target dataset
    :rtype: float
    """
    df = pd.read_pickle('jd_lst_map_idr2.pkl') # df of JD and LAST information
    lst = df.loc[df['JD_time'] == JD_time]['LASTs'].values[0][tint] # take 1st LAST
    tgt = lst_to_jd_time(lst, JD_day, telescope='HERA')
    tgt_jd_time, _ = find_nearest(df['JD_time'].values, tgt, condition='leq')
    return tgt_jd_time


def split_rel_results(resx, no_unq_bls, coords='cartesian'):
    """Split the real results array from relative calibration minimization into
    complex visibility and gains arrays

    :param resx: Optimization result for the solved antenna gains and true sky
    visibilities
    :type resx: ndarray
    :param no_unq_bls: Number of unique baselines (equivalently the number of
    redundant visibilities)
    :type no_unq_bls: int

    :return: Tuple of complex visibility and gain solution arrays
    :rtype: tuple
    """
    cfun = {'cartesian':makeCArray, 'polar':makeEArray}
    vis_params, gains_params = numpy.split(resx, [no_unq_bls*2,])
    res_vis = cfun[coords](vis_params)
    res_gains = cfun[coords](gains_params)
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


def mod_str_arg(str_arg):
    """Returns int, ndarray or None, which is readable by group_data

    :param str_arg: Number or range to select in format: '50', or '50~60'. None
    to select all.
    :type str_arg: str, None

    :return: int, ndarray or None
    :rtype: int, ndarray, None
    """
    if str_arg is not None:
        out = list(map(int, str_arg.split('~')))
        if len(out) > 1:
            out = numpy.arange(out[0], out[1]+1)
    else:
        out = None
    return out


def new_fn(out, jd_time, dt):
    """Write a new file labelled by the JD time under consideration and the
    current datetime

    :param out: Existing outfile to not overwrite
    :type out: str
    :param jd_time: Fractional JD time of dataset
    :type jd_time: float
    :param dt: Datetime to use in filename
    :type dt: datetime

    :return: New unique filename with JD time and current datetime
    :rtype:
    """
    bn = os.path.splitext(out)[0]
    ext = os.path.splitext(out)[-1]
    dt = dt.strftime('%Y_%m_%d.%H_%M_%S')
    if jd_time is None:
        jd_time = ''
    if dt is None:
        dt = datetime.datetime.now()
    out = '{}.{}.{}.{}'.format(bn, jd_time, dt, ext)
    return re.sub(r'\.+', '.', out)
