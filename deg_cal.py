"""Batch degenerate parameter fitting between relatively calibrated visibilities
across LAST, frequency, or JD

example run:
$ python deg_cal.py 2458098.43869 --deg_dim freq --pol ee --chans 300~301 --tints 10~11

Can then read the results dataframe with:
> pd.read_pickle('deg_df.2458098.43869.ee.freq.pkl')

Note that default is to write all solutions to the same csv file, for each
visibility dataset
"""


import argparse
import datetime
import io
import os
import pickle
import textwrap
from contextlib import redirect_stdout
from csv import DictWriter

import pandas as pd
import numpy

from red_likelihood import doDegVisVis, group_data
from red_utils import find_flag_file, find_rel_df, find_zen_file, fn_format, \
get_bad_ants, mod_str_arg, new_fn, split_rel_results


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.\
    RawDescriptionHelpFormatter, description=textwrap.dedent("""
    Degenerate fitting of relatively calibrated visibility solutions

    Takes the relatively calibrated visibility solutions of HERA data, and fits
    degenerate parameters (overall amplitude, overall phase, and phase gradient)
    to two adjacent datasets in either LAST, frequency, or JD.

    Returns a pickled pandas dataframe of the Scipy optimization results for
    the degenerate fitting.
    """))
    parser.add_argument('jd_time', help='Fractional JD time of dataset to \
                        analyze', metavar='JD')
    parser.add_argument('-o', '--out', required=False, default=None, \
                        metavar='O', type=str, help='Output csv and df name')
    parser.add_argument('-p', '--pol', required=True, metavar='P', type=str, \
                        help='Polarization {"ee", "en", "nn", "ne"}')
    parser.add_argument('-d', '--deg_dim', required=True, metavar='D', type=str, \
                        help='Which dimension to compare relatively calibrated \
                        visibility solutions {"last", "freq", "jd"}')
    parser.add_argument('-c', '--chans', required=False, default=None, metavar='C', \
                        type=str, help='Frequency channels to fit {0, 1023}')
    parser.add_argument('-t', '--tints', required=False, default=None, metavar='T', \
                        type=str, help='Time integrations to fit {0, 59}')
    parser.add_argument('-f', '--dist', required=False, default='cauchy', metavar='F', \
                        type=str, help='Fitting distribution for calibration \
                        {"cauchy", "gaussian"}')
    parser.add_argument('-n', '--new_csv', required=False, action='store_true', \
                        help='Write data to a new csv file')
    args = parser.parse_args()

    startTime = datetime.datetime.now()

    out_fn = args.out
    if out_fn is None:
        out_fn = 'deg_df.{}.{}.{}'.format(args.jd_time, args.pol, args.deg_dim)

    out_csv = fn_format(out_fn, 'csv')
    csv_exists = os.path.exists(out_csv)
    if csv_exists:
        if args.new_csv:
            out_csv = new_fn(out_csv, None, startTime)
            csv_exists = False

    freq_chans = mod_str_arg(args.chans)
    time_ints = mod_str_arg(args.tints)

    pchans = args.chans
    if pchans is None:
        pchans = '0~1023'
    ptints = args.tints
    if ptints is None:
        ptints = '0~59'
    print('Running degenerate fitting on visibility dataset {} for frequency '\
          'channel(s) {} and time integration(s) {}\n'.\
          format(os.path.basename(find_zen_file(args.jd_time)), pchans, ptints))

    rel_df_path = find_rel_df(args.jd_time, args.pol)
    rel_df = pd.read_pickle(rel_df_path)

    if freq_chans is None:
        freq_chans = numpy.arange(cData.shape[0])
    if time_ints is None:
        time_ints = numpy.arange(cData.shape[1])

    # filter by specified channels and time integrations
    freq_flt = numpy.in1d(rel_df.index.get_level_values('freq'), freq_chans)
    tint_flt = numpy.in1d(rel_df.index.get_level_values('time_int'), time_ints)
    rel_df = rel_df[freq_flt & tint_flt]

    with open(rel_df_path.rsplit('.', 1)[0] + '.md.pkl', 'rb') as f:
        md = pickle.load(f)
    antpos = md['antpos']
    no_unq_bls = md['no_unq_bls']
    redg = md['redg']

    indices = ['freq1', 'freq2', 'time_int']
    # not keeping 'jac', 'hess_inv', 'nfev', 'njev'
    slct_keys = ['success', 'status','message', 'fun', 'nit', 'x']
    no_deg_params = 4
    header = slct_keys[:-1] + list(numpy.arange(no_deg_params)) + indices

    # what to iterate over (pairs of datasets)
    # try for just channel comparison herewith (extend after)
    iter_dims = [idim for idim in zip(freq_chans, freq_chans[1:])] # freq pairs
    iter_dims = [idim+(time_int,) for idim in iter_dims for time_int in \
                 time_ints] # time integrations

    if csv_exists:
        # skipping freqs and tints that are already in csv file
        df = pd.read_csv(out_csv, usecols=indices)
        iter_dims = [idim for idim in iter_dims if idim not in df.values]
        if not any(iter_dims):
            print('Solutions to all specified frequency channels and time '\
                  'integrations already exist in {}\n'.format(out_csv))
            skip_cal = True
        else:
            skip_cal = False

    if not skip_cal:
        stdout = io.StringIO()
        with redirect_stdout(stdout): # suppress output
            with open(out_csv, 'a') as f: # write / append to csv file
                writer = DictWriter(f, fieldnames=header)
                if not csv_exists:
                    writer.writeheader()
                initp = None
                for iter_dim in iter_dims:
                    # get relatively calibrated solutions
                    resx1 = rel_df.loc[iter_dim[0], iter_dim[2]][len(slct_keys)-1:]\
                    .values.astype(float)
                    resx2 = rel_df.loc[iter_dim[1], iter_dim[2]][len(slct_keys)-1:]\
                    .values.astype(float)
                    rel_vis1, _ = split_rel_results(resx1, no_unq_bls)
                    rel_vis2, _ = split_rel_results(resx2, no_unq_bls)

                    res_deg = doDegVisVis(redg, antpos, rel_vis1, rel_vis2, \
                                          distribution=args.dist)
                    res_deg = {key:res_deg[key] for key in slct_keys}
                    # expanding out the solution
                    for i, param in enumerate(res_deg['x']):
                        res_deg[i] = param
                    # use solution for next solve in iteration
                    if res_deg['success']:
                        initp = res_deg['x']
                    del res_deg['x']
                    res_deg.update({indices[i]:iter_dim[i] for i in range(3)})
                    writer.writerow(res_deg) # writing to csv

        print('Degenerate fitting results saved to csv file {}'.format(out_csv))
        df = pd.read_csv(out_csv)
        df.set_index(indices, inplace=True)
        df.sort_values(by=indices, inplace=True)
        out_df = out_csv.rsplit('.', 1)[0] + '.pkl'
        df.to_pickle(out_df)
        print('Degenerate fitting results dataframe pickled to {}'.format(out_df))

    print('Script run time: {}'.format(datetime.datetime.now() - startTime))


if __name__ == '__main__':
    main()
