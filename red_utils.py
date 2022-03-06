"""Utility functions for robust redundant calibration"""


import datetime
import glob
import os
import pickle
import re
import warnings

import numpy
import pandas as pd
from scipy.optimize import minimize_scalar

import pyuvdata
from hera_cal.io import HERACal, HERAData
from hera_cal.redcal import get_reds
from pyuvdata import utils as uvutils

from red_likelihood import fltBad, groupBls, makeCArray, makeEArray

BADANTSPATH = os.path.join(os.path.dirname(__file__), 'bad_ants_idr2.pkl')
JD2LSTPATH = os.path.join(os.path.dirname(__file__), 'jd_lst_map_idr2.pkl')

warnings.filterwarnings('ignore', \
    message='telescope_location is not set. Using known values for HERA.')
warnings.filterwarnings('ignore', \
    message='antenna_positions is not set. Using known values for HERA.')


def find_zen_file(JD_time):
    """Returns visibility dataset path for the specified JD_time

    :param JD_time: Fractional Julian date
    :type JD_time: float, str

    :return: File path of visibility dataset
    :rtype: str
    """
    mdm_dir = '/Users/matyasmolnar/Downloads/HERA_Data/sample_data'
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
    :type JD_time: float, str
    :param cal_type: Calibration process that produced the flag file {"first",
    "omni", "abs", "flagged_abs", "smooth_abs"}, to name a few
    :type cal_type: str

    :return: File path of visibility dataset
    :rtype: str, None
    """
    mdm_dir = '/Users/matyasmolnar/Downloads/HERA_Data/sample_data'
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


def find_rel_df(JD_time, pol, ndist, dir=None):
    """Returns relative calibration results dataframe path for the specified
    JD time, polarization and noise distribution

    :param JD_time: Fractional Julian date
    :type JD_time: float, str
    :param pol: Polarization of data
    :type pol: str
    :param ndist: Noise distribution for calibration {"cauchy", "gaussian"}
    :type ndist: str
    :param dir: Directory in which dataframes are located
    :type dir: str

    :return: File path of relative calibration results dataframe
    :rtype: str
    """
    dir_path = '.'
    if dir is not None:
        dir_path = dir
    df_path = '{}/rel_df.{}.{}.{}.pkl'.format(dir_path, JD_time, pol, ndist)
    if not os.path.exists(df_path):
        df_glob = glob.glob('.*.'.join(df_path.rsplit('.', 1)))
        if not df_glob:
            raise ValueError('DataFrame {} not found'.format(df_path))
        else:
            df_glob.sort(reverse=True)
            df_path = df_glob[0] # get latest result as default
    return df_path


def find_deg_df(JD_time, pol, deg_dim, ndist, dir=None):
    """Returns degenerate fitting results dataframe path for the specified
    JD time, polarization, degenerate dimension and noise distribution

    :param JD_time: Fractional Julian date
    :type JD_time: float, str
    :param pol: Polarization of data
    :type pol: str
    :param deg_dim: Dimension to compare relatively calibrated visibility
    solutions {"tint", "freq", "jd"}. If "jd" specified, add JD day of 2nd
    dataset being compared, otherwise the next JD day will be assumed - e.g.
    jd.2458099
    :type deg_dim: str
    :param ndist: Noise distribution for calibration {"cauchy", "gaussian"}
    :type ndist: str
    :param dir: Directory in which dataframes are located
    :type dir: str

    :return: File path of degenerate fitting results dataframe
    :rtype: str
    """
    dir_path = '.'
    if dir is not None:
        dir_path = dir
    if deg_dim == 'jd':
        deg_dim = 'jd.{}'.format(int(float(JD_time)) + 1)
    df_path = '{}/deg_df.{}.{}.{}.{}.pkl'.format(dir_path, JD_time, pol, deg_dim, \
                                                 ndist)
    if not os.path.exists(df_path):
        df_glob = glob.glob('.*.'.join(df_path.rsplit('.', 1)))
        if not df_glob:
            raise ValueError('DataFrame {} not found'.format(df_path))
        else:
            df_glob.sort(reverse=True)
            df_path = df_glob[0] # get latest result as default
    return df_path


def find_opt_df(JD_time, pol, ndist, dir=None):
    """Returns optimal calibration results dataframe path for the specified
    JD time, polarization and noise distribution

    :param JD_time: Fractional Julian date
    :type JD_time: float, str
    :param pol: Polarization of data
    :type pol: str
    :param ndist: Noise distribution for calibration {"cauchy", "gaussian"}
    :type ndist: str
    :param dir: Directory in which dataframes are located
    :type dir: str

    :return: File path of relative calibration results dataframe
    :rtype: str
    """
    dir_path = '.'
    if dir is not None:
        dir_path = dir
    df_path = '{}/opt_df.{}.{}.{}.pkl'.format(dir_path, JD_time, pol, ndist)
    if not os.path.exists(df_path):
        df_glob = glob.glob('.*.'.join(df_path.rsplit('.', 1)))
        if not df_glob:
            raise ValueError('DataFrame {} not found'.format(df_path))
        else:
            df_glob.sort(reverse=True)
            df_path = df_glob[0] # get latest result as default
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
    with open(BADANTSPATH, 'rb') as f:
        bad_ants_dict = pickle.load(f)
    return bad_ants_dict[jd_day]


def flt_ant_coords(jd_time, antcoords, add_bad_ants=None):
    """Sort antenna coordinates according to antenna number and filter out bad
    antennas

    :param JD_time: Fractional Julian date associated with antcoords
    :type JD_time: float, str
    :param antcoords: Antenna coordinates (e.g. position, separation etc.)
    :type antcoords: dict
    :param add_bad_ants: Additional bad antennas
    :type add_bad_ants: None, int, list, ndarray

    :return: Sorted and filtered antenna coordinates
    :rtype: dict
    """
    bad_ants = get_bad_ants(find_zen_file(jd_time))
    if add_bad_ants is not None:
        bad_ants = numpy.sort(numpy.append(bad_ants, numpy.array(add_bad_ants)))
    antcoords = {ant_no: coord for ant_no, coord in sorted(antcoords.items(), \
                key=lambda item: item[0]) if ant_no not in bad_ants}
    return antcoords


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
    :type JD: float, str
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
    if isinstance(JD_time, str):
        JD_time = float(JD_time)
    df = pd.read_pickle(JD2LSTPATH) # df of JD and LAST information
    lst = df.loc[df['JD_time'] == JD_time]['LASTs'].values[0][tint] # take 1st LAST
    tgt = lst_to_jd_time(lst, JD_day, telescope='HERA')
    tgt_jd_time, _ = find_nearest(df['JD_time'].values, tgt, condition='leq')
    return tgt_jd_time


def split_opt_results(optx, no_ants):
    """Split the real results array from optimal absolute calibration minimization
    into complex visibility, degenerate parameter and complex gain arrays

    :param optx: Optimization result for optimal absolute calibration
    :type optx: ndarray
    :param no_ants: Number of antennas
    :type no_ants: int

    :return: Tuple of gain, degenerate parameters and visibility solution arrays
    :rtype: tuple
    """
    no_deg_params = 4 # overall amplitude, overall phase, tilt shifts
    gains_comps, opt_degp, vis_comps = numpy.split(optx, \
                                       [2*no_ants, 2*no_ants+no_deg_params,])
    opt_gains = makeEArray(gains_comps)
    opt_vis = makeEArray(vis_comps)
    return opt_gains, opt_degp, opt_vis


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
            out = numpy.array(out)
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
    if jd_time is None:
        jd_time = ''
    if dt is None:
        dt = datetime.datetime.now()
    dt = dt.strftime('%Y_%m_%d.%H_%M_%S')
    out = '{}.{}.{}.{}'.format(bn, jd_time, dt, ext)
    return re.sub(r'\.+', '.', out)


def check_jdt(JD_time):
    """Check that JD_time has trailing zero so that dataset can be found
    :param JD_time: JD_time that labels dataset
    :type JD_time: str, float

    :return: Checked JD_time that correctly labels dataset
    :rtype: str
    """
    if len(str(JD_time)) < 13:
        # add a trailing 0 that is omitted in float
        JD_time = str(JD_time) + '0'
    return JD_time


def calfits_to_flags(JD_time, cal_type, pol='ee', add_bad_ants=None):
    """Returns flags array from calfits file

    :param JD_time: Fractional Julian date
    :type JD_time: float, str
    :param cal_type: Calibration process that produced the calfits file {"first",
    "omni", "abs", "flagged_abs", "smooth_abs"}
    :type cal_type: str
    :param pol: Polarization of data
    :type pol: str
    :param add_bad_ants: Additional bad antennas
    :type add_bad_ants: None, int, list, ndarray

    :return: Flags array
    :rtype: ndarray
    """

    zen_fn = find_zen_file(JD_time)
    flags_fn = find_flag_file(JD_time, cal_type)
    bad_ants = get_bad_ants(zen_fn)
    if add_bad_ants is not None:
        bad_ants = numpy.sort(numpy.append(bad_ants, numpy.array(add_bad_ants)))

    hc = HERACal(flags_fn)
    _, cal_flags, _, _ = hc.read()

    hd = HERAData(zen_fn)
    reds = get_reds(hd.antpos, pols=[pol])
    reds = fltBad(reds, bad_ants)
    redg = groupBls(reds)

    antpairs = redg[:, 1:]
    cflag = numpy.empty((hd.Nfreqs, hd.Ntimes, redg.shape[0]), dtype=bool)
    for g in range(redg.shape[0]):
        cflag[:, :, g] = cal_flags[(int(antpairs[g, 0]), 'J{}'.format(pol)) or \
                                   (int(antpairs[g, 1]), 'J{}'.format(pol))].transpose()

    return cflag


def bad_ants_from_calfits(JD_time, cal_type, pol='ee'):
    """Returns bad antennas from calfits file

    :param JD_time: Fractional Julian date
    :type JD_time: float, str
    :param cal_type: Calibration process that produced the calfits file {"first",
    "omni", "abs", "flagged_abs", "smooth_abs"}
    :type cal_type: str
    :param pol: Polarization of data
    :type pol: str

    :return: Bad antennas
    :rtype: ndarray
    """
    flags_fn = find_flag_file(JD_time, cal_type)

    hc = HERACal(flags_fn)
    _, flags, _, _ = hc.read()

    bad_ants = []
    for k, v in flags.items():
        if k[1] == 'J{}'.format(pol):
            check = v.all()
            if check:
                bad_ants.append(k[0])
    return numpy.asarray(bad_ants)
