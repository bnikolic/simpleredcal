"""Batch degenerate parameter fitting between relatively calibrated visibilities
across LAST, frequency, or JD

example run:
$ python deg_cal.py 2458098.43869 --deg_dim freq --pol ee --chans 300~301 --tints 10~11

Can then read the results dataframe with:
> pd.read_pickle('deg_df.2458098.43869.ee.freq.cauchy.pkl')

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

from red_likelihood import doDegVisVis
from red_utils import find_nearest, find_rel_df, find_zen_file, fn_format, \
lst_to_jd_time, match_lst, mod_str_arg, new_fn, split_rel_results


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
                        analyze', metavar='JD', type=float)
    parser.add_argument('-o', '--out', required=False, default=None, \
                        metavar='O', type=str, help='Output csv and df name')
    parser.add_argument('-p', '--pol', required=True, metavar='P', type=str, \
                        help='Polarization {"ee", "en", "nn", "ne"}')
    parser.add_argument('-d', '--deg_dim', required=True, metavar='D', type=str, \
                        help='Which dimension to compare relatively calibrated \
                        visibility solutions {"tint", "freq", "jd"}')
    parser.add_argument('-c', '--chans', required=False, default=None, metavar='C', \
                        type=str, help='Frequency channels to fit {0, 1023}')
    parser.add_argument('-t', '--tints', required=False, default=None, metavar='T', \
                        type=str, help='Time integrations to fit {0, 59}')
    parser.add_argument('-m', '--dist', required=False, default='cauchy', metavar='F', \
                        type=str, help='Fitting distribution for calibration \
                        {"cauchy", "gaussian"}')
    parser.add_argument('-j', '--tgt_jd', required=False, default=None, metavar='J', \
                        type=float, help='JD day for fitting across JDs - only if \
                        deg_dim = "jd". Default to pick consecutive JD day')
    parser.add_argument('-n', '--new_csv', required=False, action='store_true', \
                        help='Write data to a new csv file')
    args = parser.parse_args()

    startTime = datetime.datetime.now()

    pjd = ''
    if args.deg_dim == 'jd':
        tgt_jd = args.tgt_jd
        if tgt_jd is None:
            tgt_jd = int(args.jd_time) + 1 # choose consecutive JD
        pjd = '.' + str(tgt_jd)

    out_fn = args.out
    if out_fn is None:
        out_fn = 'deg_df.{}.{}.{}{}.{}'.format(args.jd_time, args.pol, \
                                               args.deg_dim, pjd, args.dist)

    out_csv = fn_format(out_fn, 'csv')
    csv_exists = os.path.exists(out_csv)
    if csv_exists:
        if args.new_csv:
            out_csv = new_fn(out_csv, None, startTime)
            csv_exists = False

    freq_chans = mod_str_arg(args.chans)
    time_ints = mod_str_arg(args.tints)

    rel_df_path = find_rel_df(args.jd_time, args.pol, args.dist)
    rel_df = pd.read_pickle(rel_df_path)

    # retrieving visibility metadata
    with open(rel_df_path.rsplit('.', 2)[0] + '.md.pkl', 'rb') as f:
        md = pickle.load(f)
    antpos = md['antpos']
    no_unq_bls = md['no_unq_bls']
    redg = md['redg']

    pchans = args.chans
    if pchans is None:
        pchans = '0~{}'.format(md['Nfreqs']-1)
    ptints = args.tints
    if ptints is None:
        ptints = '0~{}'.format(md['Ntimes']-1)
    pdict = {'freq':'frequency channels', 'tint':'time integrations', \
             'jd':'Julian days'}
    print('Running degenerate fitting on adjacent {} for visibility dataset {} '\
          'for frequency channel(s) {} and time integration(s) {}\n'.\
          format(pdict[args.deg_dim], os.path.basename(find_zen_file(args.jd_time)), \
          pchans, ptints))

    if freq_chans is None:
        freq_chans = numpy.arange(md['Nfreqs'])
    if time_ints is None:
        time_ints = numpy.arange(md['Ntimes'])

    # filter by specified channels and time integrations
    freq_flt = numpy.in1d(rel_df.index.get_level_values('freq'), freq_chans)
    tint_flt = numpy.in1d(rel_df.index.get_level_values('time_int'), time_ints)
    rel_df = rel_df[freq_flt & tint_flt]

    # only getting frequencies and time integrations that exist in the df
    freq_chans = rel_df.index.get_level_values('freq').unique().values
    time_ints = rel_df.index.get_level_values('time_int').unique().values

    if args.deg_dim == 'freq':
        indices = ['freq1', 'freq2', 'time_int']
        # getting adjacent frequency channel pairs
        iter_dims = [idim for idim in zip(freq_chans, freq_chans[1:]) if \
                     idim[1] - idim[0] == 1]
        iter_dims = [idim+(time_int,) for idim in iter_dims for time_int in \
                     time_ints] # adding time integrations
        a, b, c, d = 0, 1, 2, 2 # for iteration indexing

    if args.deg_dim == 'tint':
        indices = ['time_int1', 'time_int2', 'freq']
        # getting adjacent LAST (time integration) pairs
        iter_dims = [idim for idim in zip(time_ints, time_ints[1:]) if \
                     idim[1] - idim[0] == 1]
        iter_dims = [idim+(freq_chan,) for idim in iter_dims for freq_chan in \
                     freq_chans] # adding frequency channels
        a, b, c, d = 2, 2, 0, 1 # for iteration indexing

    rel_df_c = rel_df
    if args.deg_dim == 'jd':
        indices = ['time_int1', 'time_int2', 'freq']
        # find dataset from specified JD that contains visibilities at the same LAST
        jd_time2 = match_lst(args.jd_time, tgt_jd)
        rel_df_path2 = find_rel_df(jd_time2, args.pol, args.dist)
        # aligning datasets in LAST
        last_df = pd.read_pickle('jd_lst_map_idr2.pkl')
        last1 = last_df[last_df['JD_time'] == args.jd_time]['LASTs'].values[0]
        last2 = last_df[last_df['JD_time'] == jd_time2]['LASTs'].values[0]
        _, offset = find_nearest(last2, last1[0])

        rel_df2 = pd.read_pickle(rel_df_path2)
        rel_df2 = rel_df2[rel_df2.index.get_level_values('time_int') >= offset]

        next_row = numpy.where(last_df['JD_time'] == jd_time2)[0][0] + 1
        rel_df_path3 = find_rel_df(last_df.iloc[next_row]['JD_time'], args.pol, \
                                   args.dist)
        rel_df3 = pd.read_pickle(rel_df_path3)
        rel_df3 = rel_df3[rel_df3.index.get_level_values('time_int') < offset]

        # combined results dataframes that is now alinged in LAST by row number
        # with rel_df:
        rel_df_c = pd.concat([rel_df2, rel_df3])
        # pairing time_ints from rel_df and rel_df_c that match in LAST
        time_ints2 = rel_df_c.index.get_level_values('time_int').unique().values
        iter_dims = [idim for idim in zip(time_ints, time_ints2)]
        iter_dims = [idim+(freq_chan,) for idim in iter_dims for freq_chan in \
                     freq_chans]
        iter_dims = sorted(iter_dims, key=lambda row: row[2]) # iterate across
        # LAST first - should speed up fitting
        a, b, c, d = 2, 2, 0, 1 # for iteration indexing

    # not keeping 'jac', 'hess_inv', 'nfev', 'njev'
    slct_keys = ['success', 'status','message', 'fun', 'nit', 'x']
    no_deg_params = 4
    header = slct_keys[:-1] + list(numpy.arange(no_deg_params)) + indices

    skip_cal = False
    if csv_exists:
        # skipping freqs and tints that are already in csv file
        df = pd.read_csv(out_csv, usecols=indices)
        iter_dims = [idim for idim in iter_dims if not \
            numpy.equal(df.values, numpy.asarray(idim)).all(1).any()]
        if not any(iter_dims):
            print('Solutions to all specified frequency channels and time '\
                  'integrations already exist in {}\n'.format(out_csv))
            skip_cal = True

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
                    resx1 = rel_df.loc[iter_dim[a], iter_dim[c]][len(slct_keys)-1:]\
                    .values.astype(float)
                    resx2 = rel_df_c.loc[iter_dim[b], iter_dim[d]][len(slct_keys)-1:]\
                    .values.astype(float)
                    rel_vis1, _ = split_rel_results(resx1, no_unq_bls)
                    rel_vis2, _ = split_rel_results(resx2, no_unq_bls)

                    res_deg = doDegVisVis(redg, antpos, rel_vis1, rel_vis2, \
                                          distribution=args.dist, initp=initp)
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
