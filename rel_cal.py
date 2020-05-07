"""Batch relative redundant calibration of visibilities across frequencies
and time

example run:
$ python rel_cal.py 2458098.43869 --pol 'ee' --chans 300~301 --tints 0~1 --overwrite

Can then read the dataframe with:
> pd.read_pickle('res_df.pkl')

TODO:
- Iteratively appending rows to a DataFrame can be more computationally
intensive than a single concatenate. A better solution is to append those rows
to a list and then concatenate the list with the original DataFrame all at once.
    -> do this every frequency channel?

- Writing dataframe to disk every nth frequency iteration
"""


import argparse
import datetime
import io
import os
import textwrap
from contextlib import redirect_stdout

import pandas as pd
import numpy

from red_likelihood import doRelCal, group_data
from red_utils import find_zen_file, fn_format, get_bad_ants


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
    parser.add_argument('--out_df', required=False, default='res_df.pkl', \
                        metavar='O', type=str, help='Output dataframe name')
    parser.add_argument('--pol', required=True, metavar='pol', type=str, \
                        help='Polarization {"ee", "en", "nn", "ne"}')
    parser.add_argument('--chans', required=False, default=None, metavar='C', \
                        type=str, help='Frequency channels to calibrate \
                        {0, 1023}')
    parser.add_argument('--tints', required=False, default=None, metavar='T', \
                        type=str, help='Time integrations to calibrate \
                        {0, 59}')
    parser.add_argument('--overwrite', required=False, action='store_true', \
                        help='Overwrite existing dataframe')
    args = parser.parse_args()

    startTime = datetime.datetime.now()

    ext = 'pkl'
    out_df = fn_format(args.out_df, ext)
    if os.path.exists(out_df):
        if not args.overwrite:
            bn = os.path.splitext(out_df)[0]
            dt = startTime.strftime('%Y_%m_%d.%H_%M_%S')
            out_df = '{}.{}.{}.{}'.format(bn, args.jd_time, dt, ext)

    filename = find_zen_file(args.jd_time)
    bad_ants = get_bad_ants(filename)
    pol = args.pol
    freq_chans = mod_str_arg(args.chans)
    time_ints = mod_str_arg(args.tints)

    hdraw, cRedG, cData = group_data(filename, pol, freq_chans, bad_ants)

    if freq_chans is None:
        freq_chans = numpy.arange(cData.shape[0])
    if time_ints is None:
        time_ints = numpy.arange(cData.shape[1])

    indices = ['freq', 'time_int']
    # not keeping 'jac', 'hess_inv', 'nfev', 'njev'
    slct_keys = ['success', 'status','message', 'fun', 'nit', 'x']
    df = pd.DataFrame(columns=indices+slct_keys)
    stdout = io.StringIO()
    with redirect_stdout(stdout): # suppress output
        initp = None
        for iter_dim in numpy.ndindex(cData[:, time_ints, :].shape[:2]):
            res_rel = doRelCal(cRedG, cData[iter_dim], distribution='cauchy', \
                               initp=initp)
            res_rel = {key:res_rel[key] for key in slct_keys}
            res_rel.update({indices[0]:freq_chans[iter_dim[0]], \
                            indices[1]:time_ints[iter_dim[1]]})
            df = df.append(res_rel, ignore_index=True)
            initp = res_rel['x'] # use solution for next solve in iteration

    df.set_index(indices, inplace=True)
    df.to_pickle(out_df)
    print('Relative calibration results saved to dataframe')
    print('Script run time: {}'.format(datetime.datetime.now() - startTime))


if __name__ == '__main__':
    main()
