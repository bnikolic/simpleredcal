"""Batch relative redundant calibration of visibilities across frequencies
and time

example run:
$ python rel_cal.py 2458098.43869 --pol 'ee' --chans 300~301 --tints 0~1 --flag_type first

Can then read the dataframe with:
> pd.read_pickle('res_df.2458098.43869.ee.pkl')

Note that default is to write all solutions to the same csv file, for each
visibility dataset
"""


import argparse
import datetime
import io
import os
import pickle
import re
import textwrap
from contextlib import redirect_stdout
from csv import DictWriter

import pandas as pd
import numpy

from red_likelihood import doRelCal, group_data, norm_rel_sols
from red_utils import find_flag_file, find_zen_file, fn_format, get_bad_ants


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
    parser.add_argument('-o', '--out', required=False, default=None, \
                        metavar='O', type=str, help='Output csv and df name')
    parser.add_argument('-p', '--pol', required=True, metavar='P', type=str, \
                        help='Polarization {"ee", "en", "nn", "ne"}')
    parser.add_argument('-c', '--chans', required=False, default=None, metavar='C', \
                        type=str, help='Frequency channels to calibrate \
                        {0, 1023}')
    parser.add_argument('-t', '--tints', required=False, default=None, metavar='T', \
                        type=str, help='Time integrations to calibrate \
                        {0, 59}')
    parser.add_argument('-f', '--flag_type', required=False, default=None, \
                        metavar='F', type=str, help='Flag type e.g. "first", \
                        "omni", "abs"')
    parser.add_argument('-n', '--new_csv', required=False, action='store_true', \
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

    zen_fn = find_zen_file(args.jd_time)
    bad_ants = get_bad_ants(zen_fn)

    flag_type = args.flag_type
    if flag_type is not None:
        flag_fn = find_flag_file(args.jd_time, flag_type)
    else:
        flag_fn = None

    pol = args.pol
    freq_chans = mod_str_arg(args.chans)
    time_ints = mod_str_arg(args.tints)

    pchans = args.chans
    if pchans is None:
        pchans = '0~1023'
    ptints = args.tints
    if ptints is None:
        ptints = '0~59'
    print('Running relative redundant calibration on visibility dataset {} for '\
          'polarization {}, frequency channel(s) {} and time integration(s) {}\n'.\
          format(os.path.basename(zen_fn), args.pol, pchans, ptints))

    hdraw, RedG, cData = group_data(zen_fn, pol, freq_chans, time_ints, \
                                    bad_ants, flag_path=flag_fn)
    flags = cData.mask
    cData = cData.data

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
            print('Solutions to all specified frequency channels and time '\
                  'integrations already exist in {}\n'.format(out_csv))

    # remove flagged channels from iter_dims
    if True in flags:
        flg_chans = numpy.where(flags.all(axis=(1,2)))[0] # indices
        print('Flagged channels for visibility dataset {} are: {}\n'.\
             format(os.path.basename(zen_fn), freq_chans[flg_chans]))
        iter_dims = [idim for idim in iter_dims if idim[0] not in flg_chans]

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
                res_rel['x'] = norm_rel_sols(res_rel['x'], no_unq_bls)
                # expanding out the solution
                for i, param in enumerate(res_rel['x']):
                    res_rel[i] = param
                # use solution for next solve in iteration
                if res_rel['success']:
                    initp = res_rel['x']
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

    # creating metadata file
    out_md = out_df.rsplit('.', 1)[0] + '.md.pkl'
    if not os.path.exists(out_md):
        md = {'no_ants':no_ants, 'no_unq_bls':no_unq_bls, 'redg':RedG}
        with open(out_df.rsplit('.', 1)[0] + '.md.pkl', 'wb') as f:
            pickle.dump(md, f, protocol=pickle.HIGHEST_PROTOCOL)
    print('Relative calibration metadata pickled to {}\n'.format(out_md))

    print('Script run time: {}'.format(datetime.datetime.now() - startTime))


if __name__ == '__main__':
    main()
