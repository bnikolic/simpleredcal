"""Robust redundant calibration across days"""


import os
import pickle
import sys
import warnings
from collections import OrderedDict as odict

import numpy
import pandas as pd

from hera_cal.datacontainer import DataContainer
from hera_cal.io import HERAData
from hera_cal.utils import lst_rephase

from jax.config import config
config.update('jax_enable_x64', True)
from jax import numpy as np

from red_likelihood import group_data
from red_utils import check_jdt, find_flag_file, find_nearest, find_zen_file, \
match_lst

warnings.filterwarnings('ignore', \
    message='telescope_location is not set. Using known values for HERA.')
warnings.filterwarnings('ignore', \
    message='antenna_positions is not set. Using known values for HERA.')


def union_bad_ants(JDs):
    """Return all the bad antennas for the specified JDs

    :param: Julian Days
    :type: ndarray, list

    :return: Union of bad antennas for JDs
    :rtype: ndarray
    """
    bad_ants_fn = os.path.join(os.path.dirname(__file__), 'bad_ants_idr2.pkl')
    with open(bad_ants_fn, 'rb') as f:
        bad_ants_dict = pickle.load(f)
    bad_ants = np.array([], dtype=int)
    for JD in JDs:
        bad_ants = np.append(bad_ants, bad_ants_dict[JD])
    return np.sort(np.unique(bad_ants))


def suppressOutput(func):
    """Utility function that blocks print calls by writing to
    the null device"""
    def func_wrapper(*args, **kwargs):
        with open(os.devnull, 'w') as devNull:
            old_stdout = sys.stdout
            sys.stdout = devNull
            value = func(*args, **kwargs)
            sys.stdout = old_stdout
        return value
    return func_wrapper


def XDgroup_data(JD_time, JDs, pol, chans=None, tints=None, bad_ants=True, \
                 use_flags='first', noise=False, use_cal=None, rephase=False, \
                 verbose=False):
    """Returns redundant baseline grouping and reformatted dataset, with
    external flags applied, if specified

    :param JD_time: Julian time of 1st dataset, which sets times for others
    :type JD_time: str
    :param JDs: Julian days of data
    :type JDs: list, ndarray
    :param pol: Polarization of data
    :type pol: str
    :param chans: Frequency channel(s) {0, 1023} (None to choose all)
    :type chans: array-like, int, or None
    :param tints: Time integrations {0, 59} (None to choose all)
    :type tints: array-like, int, or None
    :param bad_ants: Flag known bad antennas, optional
    :type bad_ants: bool
    :param use_flags: Use flags to mask data
    :type use_flags: str
    :param noise: Also calculate noise from autocorrelations
    :type noise: bool
    :param use_cal: calfits file extension to use to calibrate data
    :type use_cal: str, None
    :param rephase: phase data to centre of the LST bin before binning (centre
    of the bin being the mean LST for each row across JDs)
    :type rephase: bool
    :param verbose: Print data gathering steps for each dataset
    :type verbose: bool

    :return hd: HERAData class
    :rtype hd: HERAData class
    :return redg: Grouped baselines, as returned by groupBls
    :rtype redg: ndarray
    :return cdata: Grouped visibilities with flags in numpy MaskedArray format,
    with format consistent with redg and dimensions (freq chans,
    time integrations, baselines)
    :rtype cdata: MaskedArray
    """

    if isinstance(chans, int):
        chans = np.asarray([chans])
    if isinstance(tints, int):
        tints = np.asarray([tints])

    zen_fn = find_zen_file(JD_time)
    flags_fn = find_flag_file(JD_time, use_flags)

    hd = HERAData(zen_fn)
    if tints is None:
        tints = np.arange(hd.Ntimes)

    if bad_ants:
        bad_ants = union_bad_ants(JDs)
    else:
        bad_ants = None

    if use_cal is None:
        cal_path = None
    else:
        cal_path = find_flag_file(JD_time, use_cal)

    if not verbose:
        grp_data = suppressOutput(group_data)
    else:
        grp_data = group_data

    # for rephasing
    comb_lsts = numpy.empty((len(JDs), len(tints)))
    comb_lsts[0, :] = hd.lsts[numpy.asarray(tints)]

    grp = grp_data(zen_fn, pol, chans=chans, tints=tints, bad_ants=bad_ants,
                   flag_path=flags_fn, noise=noise, cal_path=cal_path)
    _, redg, cMData = grp[:3]

    cMData = cMData[np.newaxis, :]
    if noise:
        cNoise = grp[3]
        cNoise = cNoise[np.newaxis, :]

    JD_day = int(float(JD_time))
    if JD_day in JDs:
        JDs = list(JDs)
        JDs_arr = numpy.array(JDs)
        JDs.remove(JD_day)

    for jd_i in JDs:
        JD_time_ia = match_lst(JD_time, jd_i)
        # aligning datasets in LAST
        last_df = pd.read_pickle(os.path.join(os.path.dirname(__file__), 'jd_lst_map_idr2.pkl'))
        last1 = last_df[last_df['JD_time'] == float(JD_time)]['LASTs'].values[0]
        last2 = last_df[last_df['JD_time'] == float(JD_time_ia)]['LASTs'].values[0]
        _, offset = find_nearest(last2, last1[0])
        tints_i = (tints + offset)%60
        scnd_dataset = all(tints+offset > hd.Ntimes-1)
        single_dataset = all(tints+offset < hd.Ntimes-1) or scnd_dataset

        if not single_dataset:
            tints_ia, tints_ib = np.split(tints_i, np.where(tints_i == 0)[0])
        else:
            tints_ia = tints_i

        if scnd_dataset:
            next_row = numpy.where(last_df['JD_time'] == float(JD_time_ia))[0][0] + 1
            JD_time_ib = last_df.iloc[next_row]['JD_time']
            JD_time_ia = JD_time_ib

        JD_time_ia = check_jdt(JD_time_ia)
        zen_fn_ia = find_zen_file(JD_time_ia)
        flags_fn_ia = find_flag_file(JD_time_ia, use_flags)
        if use_cal is not None:
            cal_path_ia = find_flag_file(JD_time_ia, use_cal)
        else:
            cal_path_ia = None
        grp_a = grp_data(zen_fn_ia, pol, chans=chans, tints=tints_ia, \
                         bad_ants=bad_ants, flag_path=flags_fn_ia, noise=noise, \
                         cal_path=cal_path_ia)
        cMData_ia = grp_a[2]

        lsts_i = grp_a[0].lsts[numpy.asarray(tints_ia)]

        if not single_dataset:
            next_row = numpy.where(last_df['JD_time'] == float(JD_time_ia))[0][0] + 1
            JD_time_ib = last_df.iloc[next_row]['JD_time']
            JD_time_ib = check_jdt(JD_time_ib)
            zen_fn_ib = find_zen_file(JD_time_ib)
            flags_fn_ib = find_flag_file(JD_time_ib, use_flags)
            if use_cal is not None:
                cal_path_ib = find_flag_file(JD_time_ib, use_cal)
            else:
                cal_path_ib = None
            grp_b = grp_data(zen_fn_ib, pol, chans=chans, tints=tints_ib, \
                             bad_ants=bad_ants, flag_path=flags_fn_ib, \
                             noise=noise, cal_path=cal_path_ib)
            cMData_ib = grp_b[2]

            lsts_i = numpy.append(lsts_i, grp_b[0].lsts[numpy.asarray(tints_ib)])

            cMData_i = numpy.ma.concatenate((cMData_ia, cMData_ib), axis=1)
        else:
            cMData_i = cMData_ia

        comb_lsts[JDs.index(jd_i)+1, :] = lsts_i

        cMData_i = cMData_i[np.newaxis, :]
        cMData = numpy.ma.concatenate((cMData, cMData_i), axis=0)

        if noise:
            cNoise_ia = grp_a[3]
            if not single_dataset:
                cNoise_ib = grp_b[3]
                cNoise_i =  np.concatenate((cNoise_ia, cNoise_ib), axis=1)
            else:
                cNoise_i = cNoise_ia
            cNoise_i = cNoise_i[np.newaxis, :]
            cNoise = np.concatenate((cNoise, cNoise_i), axis=0)

    if rephase:
        lst_bin_centres = np.mean(comb_lsts, axis=0)

        freq_arr = hd.freqs[numpy.asarray(chans)]
        cData_rph = numpy.empty((JDs_arr.size, freq_arr.size, len(tints), redg.shape[0]), \
                                dtype=complex)

        # get rephased data for each day
        for jd_idx, jd_i in enumerate(JDs_arr):
            # convert to DataContainer to feed into lst_rephase hera_cal function
            data_cont = odict()
            for bl_idx, bl in enumerate(redg[:, 1:]):
                data_cont[(bl[0], bl[1], pol)] = cMData.data[jd_idx, ..., bl_idx].transpose()
            data_cont = DataContainer(data_cont)

            if jd_idx == 0:
                # only need to compute once since ant positions fixed
                bls = odict([(k, hd.antpos[k[0]] - hd.antpos[k[1]]) for k in data_cont.keys()])

            # lst deltas to bin centre of aligned array
            dlst = numpy.asarray(lst_bin_centres - comb_lsts[jd_idx, :])

            # rephase data here
            data_cont = lst_rephase(data_cont, bls, freq_arr, dlst, inplace=False, array=False)

            for bl_idx, bl in enumerate(redg[:, 1:]):
                cData_rph[jd_idx, ..., bl_idx] = data_cont[(bl[0], bl[1], pol)].transpose()

        cMData = numpy.ma.masked_array(cData_rph, mask=cMData.mask, fill_value=np.nan)

    if noise:
        return hd, redg, cMData, cNoise
    else:
        return hd, redg, cMData
