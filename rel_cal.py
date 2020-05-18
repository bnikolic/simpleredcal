"""Batch relative redundant calibration of visibilities across frequencies
and time

example run:
$ python rel_cal.py 2458098.43869 --pol 'ee' --chans 300~301 --tints 0~1

Can then read the dataframe with:
> pd.read_pickle('res_df.2458098.43869.ee.pkl')

Note that default is to write all solutions to the same csv file, for each
visibility dataset
"""


import argparse
import datetime
import io
import os
import re
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
    parser.add_argument('--out', required=False, default=None, \
                        metavar='O', type=str, help='Output csv and df name')
    parser.add_argument('--pol', required=True, metavar='pol', type=str, \
                        help='Polarization {"ee", "en", "nn", "ne"}')
    parser.add_argument('--chans', required=False, default=None, metavar='C', \
                        type=str, help='Frequency channels to calibrate \
                        {0, 1023}')
    parser.add_argument('--tints', required=False, default=None, metavar='T', \
                        type=str, help='Time integrations to calibrate \
                        {0, 59}')
    parser.add_argument('--new_csv', required=False, action='store_true', \
                        help='Write data to a new csv file')
    args = parser.parse_args()

    startTime = datetime.datetime.now()

    out_fn = args.out
    if out_fn is None:
        out_fn = 'res_df.{}.{}'.format(args.jd_time, args.pol)

    out_csv = fn_format(out_fn, 'csv')
    csv_exists = os.path.exists(out_csv)
    if csv_exists:
        if args.new_csv:
            out_csv = new_fn(out_csv, None, startTime)
            csv_exists = False

    filename = find_zen_file(args.jd_time)
    bad_ants = get_bad_ants(filename)
    pol = args.pol
    freq_chans = mod_str_arg(args.chans)
    time_ints = mod_str_arg(args.tints)

    hdraw, RedG, cData = group_data(filename, pol, freq_chans, time_ints, \
                                    bad_ants, flag_path=None)
    cData = cData.data # ignore masks for the time being

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

    iter_dims = list(numpy.ndindex(cData.shape[:2]))
    if csv_exists:
        # skipping freqs and tints that are already in csv file
        df = pd.read_csv(out_csv, usecols=indices)
        cmap_f = dict(map(reversed, enumerate(freq_chans)))
        cmap_t = dict(map(reversed, enumerate(time_ints)))
        done = [(cmap_f[f], cmap_t[t]) for (f, t) in df.values if (f in freq_chans \
        and t in time_ints)]
        iter_dims = [idim for idim in iter_dims if idim not in done]
        if not any(iter_dims):
            print('Solutions to all specified frequency channels and time \
                   integrations already exist in {}'.format(out_csv))

    stdout = io.StringIO()
    with redirect_stdout(stdout): # suppress output
        with open(out_csv, 'a') as f: # write / append to csv file
            writer = DictWriter(f, fieldnames=header)
            if not csv_exists:
                writer.writeheader()
            initp = None
            for iter_dim in iter_dims:
                res_rel = doRelCal(RedG, cData[iter_dim], distribution='cauchy', \
                                   initp=initp)
                res_rel = {key:res_rel[key] for key in slct_keys}
                initp = res_rel['x'] # use solution for next solve in iteration
                for i, param in enumerate(res_rel['x']): # expanding out the solution
                    res_rel[i] = param
                del res_rel['x']
                res_rel.update({indices[0]:freq_chans[iter_dim[0]], \
                                indices[1]:time_ints[iter_dim[1]]})
                writer.writerow(res_rel) # writing to csv

    print('Relative calibration results saved to csv file {}'.format(out_csv))
    df = pd.read_csv(out_csv)
    df.set_index(indices, inplace=True)
    df.sort_values(by=indices, inplace=True)
    out_df = out_csv.rsplit('.', 1)[0] + '.pkl'
    df.to_pickle(out_df)
    print('Relative calibration results dataframe pickled to {}'.format(out_df))
    print('Script run time: {}'.format(datetime.datetime.now() - startTime))


if __name__ == '__main__':
    main()
