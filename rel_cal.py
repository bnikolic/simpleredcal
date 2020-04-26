"""Batch relative redundant calibration of visibilities across frequencies
and time"""


import argparse
import os
import textwrap

import pandas as pd
import numpy

from red_likelihood import doRelCal, group_data
from red_utils import find_zen_file, get_bad_ants


def mod_str_arg(str_arg):
    """Returns int, ndarray or None, which is readable by group_data

    :param str_arg: Number or range to select in format: '50', or '50~60'. None
    to select all.
    :type str_arg: str, None

    :return: int, ndarray or None
    :rtype: int, ndarray, None
    """
    if str_arg is not None:
        chans = list(map(int, str_arg.split('~')))
        if len(chans) > 1:
            chans = numpy.arange(chans[0], chans[1]+1)
    else:
        chans = None
    return chans


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.\
    RawDescriptionHelpFormatter, description=textwrap.dedent("""
    Relative redundant calibration of visibilities

    Takes a given HERA visibility dataset in uvh5 file format and performs
    relative redundant calibration (up to the overall ampltitude, overall
    phase, and phase gradient degenerate parameters) for each frequency channel
    and each time integration in the dataset.

    Returns a pickled pandas dataframe of the Scipy optimization results for
    the relative redundant calibration for each set of frequency channel and
    time integration.
    """))
    parser.add_argument('jd_time', help='Fractional JD time of dataset to \
                        calibrate', metavar='JD')
    parser.add_argument('--out_df', required=False, default='res_df', \
                        metavar='O', type=str, help='Output dataframe name')
    parser.add_argument('--pol', required=True, metavar='pol', type=str, \
                        help='Polarization {"ee", "en", "nn", "ne"}')
    parser.add_argument('--chans', required=False, default=None, metavar='C',
                        type=str, help='Frequency channels to calibrate \
                        {0, 1023}')
    parser.add_argument('--tints', required=False, default=None, metavar='T',
                        type=str, help='Time integrations to calibrate \
                        {0, 59}')
    args = parser.parse_args()

    filename = find_zen_file(args.jd_time)
    bad_ants = get_bad_ants(filename)
    pol = args.pol
    freq_chans = mod_str_arg(args.chans)
    time_ints = mod_str_arg(args.tints)

    hdraw, cRedG, cData = group_data(filename, pol, freq_chans, bad_ants)

    if time_ints is None:
        time_ints = numpy.arange(cData.shape[1])
    res_dict = {}
    with open(os.devnull, 'w'):
        initp = None
        for iter_dim in numpy.ndindex(cData[:, time_ints, :].shape[:2]):
            res_rel = doRelCal(cRedG, cData[iter_dim], distribution='cauchy', \
                               initp=initp)
            initp = res_rel['x'] # use solution for next solve in iteration
            res_dict[iter_dim] = res_rel

    df = pd.DataFrame.from_dict(res_dict, orient='index')
    df[['freq', 'time_int']] = pd.DataFrame(df.index.tolist(), index=df.index)
    df.reset_index(drop=True, inplace=True)
    df.set_index(['freq', 'time_int'], inplace=True)
    df.to_pickle('./{}.pkl'.format(args.out_df))
    print('Relative calibration results saved to dataframe')


if __name__ == '__main__':
    main()
