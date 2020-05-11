"""Batch relative redundant calibration of visibilities across frequencies
and time

example run:
$ python rel_cal.py 2458098.43869 --pol 'ee' --chans 300~301 --tints 0~1 \
--overwrite_df

Can then read the dataframe with:
> pd.read_pickle('res_df.pkl')

Note that default is to write all solutions to the same csv file.

TODO:
    - Additional labels for JD time and polarizations (when we move to
    processing many observations at the same time)
"""


import argparse
import datetime
import io
import os
import textwrap
from contextlib import redirect_stdout
from csv import DictWriter

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


def new_fn(out, jd_time, datetime):
    """Write a new file labelled by the JD time under consideration and the
    current datetime

    :param out: Existing outfile to not overwrite
    :type out: str
    :param jd_time: Fractional JD time of dataset
    :type jd_time: float
    :param datetime: Datetime to use in filename
    :type datetime: datetime

    :return: New unique filename with JD time and current datetime
    :rtype:
    """
    bn = os.path.splitext(out)[0]
    ext = os.path.splitext(out)[-1]
    dt = datetime.strftime('%Y_%m_%d.%H_%M_%S')
    out = '{}.{}.{}.{}'.format(bn, jd_time, dt, ext)
    return out


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
    parser.add_argument('--out', required=False, default='res_df', \
                        metavar='O', type=str, help='Output csv and df name')
    parser.add_argument('--pol', required=True, metavar='pol', type=str, \
                        help='Polarization {"ee", "en", "nn", "ne"}')
    parser.add_argument('--chans', required=False, default=None, metavar='C', \
                        type=str, help='Frequency channels to calibrate \
                        {0, 1023}')
    parser.add_argument('--tints', required=False, default=None, metavar='T', \
                        type=str, help='Time integrations to calibrate \
                        {0, 59}')
    parser.add_argument('--diff_csv', required=False, action='store_true', \
                        help='Write data to a different csv file')
    parser.add_argument('--overwrite_df', required=False, action='store_true', \
                        help='Overwrite pickled dataframe')
    args = parser.parse_args()

    startTime = datetime.datetime.now()

    out_csv = fn_format(args.out, 'csv')
    csv_exists = os.path.exists(out_csv)
    if csv_exists:
        if args.diff_csv:
            out_csv = new_fn(out_csv, args.jd_time, startTime)

    filename = find_zen_file(args.jd_time)
    bad_ants = get_bad_ants(filename)
    pol = args.pol
    freq_chans = mod_str_arg(args.chans)
    time_ints = mod_str_arg(args.tints)

    hdraw, RedG, cData = group_data(filename, pol, freq_chans, bad_ants)

    if freq_chans is None:
        freq_chans = numpy.arange(cData.shape[0])
    if time_ints is None:
        time_ints = numpy.arange(cData.shape[1])

    # to get fields for the csv header
    no_ants = numpy.unique(RedG[:, 1:]).size
    no_unq_bls = numpy.unique(RedG[:, 0]).size
    psize = (no_ants + no_unq_bls)*2

    indices = ['freq', 'time_int']
    # not keeping 'jac', 'hess_inv', 'nfev', 'njev'
    slct_keys = ['success', 'status','message', 'fun', 'nit', 'x']
    header = header = slct_keys[:-1] + list(numpy.arange(psize)) + indices
    stdout = io.StringIO()
    with redirect_stdout(stdout): # suppress output
        with open(out_csv, 'a') as f:
            writer = DictWriter(f, fieldnames=header)
            if not csv_exists:
                writer.writeheader()
            initp = None
            for iter_dim in numpy.ndindex(cData[:, time_ints, :].shape[:2]):
                res_rel = doRelCal(RedG, cData[iter_dim], distribution='cauchy', \
                                   initp=initp)
                res_rel = {key:res_rel[key] for key in slct_keys}
                initp = res_rel['x'] # use solution for next solve in iteration
                # expanding out the solution
                for i, param in enumerate(res_rel['x']):
                    res_rel[i] = param
                del res_rel['x']
                res_rel.update({indices[0]:freq_chans[iter_dim[0]], \
                                indices[1]:time_ints[iter_dim[1]]})
                writer.writerow(res_rel) # writing to csv

    print('Relative calibration results saved to csv file {}'.format(out_csv))
    df = pd.read_csv(out_csv)
    df.set_index(indices, inplace=True)
    out_df = out_csv.split('.')[0] + '.pkl'
    if os.path.exists(out_df):
        if not args.overwrite_df:
            out_df = new_fn(out_df, args.jd_time, startTime)
    df.to_pickle(out_df)
    print('Relative calibration results dataframe pickled to {}'.format(out_df))
    print('Script run time: {}'.format(datetime.datetime.now() - startTime))


if __name__ == '__main__':
    main()
