"""Batch degenerate parameter solver between relatively calibrated visibilities
across LAST, frequency, or JD

example run:
$ python deg_cal.py '2458098.43869' --deg_dim 'freq' --pol 'ee' --chans 300~301 --tints 10~11

Can then read the results dataframe with:
> pd.read_pickle('deg_df.2458098.43869.ee.freq.cauchy.pkl')

In the current setup, we do not reuse previous solutions to initialize the next
solver in the iteration, as we find that we get stuck in local minima, especially
after iterations that have bad data (due to e.g. RFI)

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

from fit_diagnostics import append_residuals_deg
from red_likelihood import doDegVisVis, red_ant_sep, split_rel_results
from red_utils import find_nearest, find_rel_df, find_zen_file, fn_format, \
match_lst, mod_str_arg, new_fn


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
                        analyze', metavar='JD', type=str)
    parser.add_argument('-o', '--out', required=False, default=None, \
                        metavar='O', type=str, help='Output csv and df name')
    parser.add_argument('-p', '--pol', required=True, metavar='P', type=str, \
                        help='Polarization {"ee", "en", "nn", "ne"}')
    parser.add_argument('-x', '--deg_dim', required=True, metavar='X', type=str, \
                        help='Which dimension to compare relatively calibrated \
                        visibility solutions {"tint", "freq", "jd"}')
    parser.add_argument('-m', '--coords', required=False, default='cartesian', \
                        metavar='M', type=str, help='Coordinates used for rel cal \
                        results - {"cartesian", "polar"}')
    parser.add_argument('-c', '--chans', required=False, default=None, metavar='C', \
                        type=str, help='Frequency channels to fit {0, 1023}')
    parser.add_argument('-t', '--tints', required=False, default=None, metavar='T', \
                        type=str, help='Time integrations to fit {0, 59}')
    parser.add_argument('-d', '--dist', required=True, default='cauchy', metavar='D', \
                        type=str, help='Fitting distribution for calibration \
                        {"cauchy", "gaussian"}')
    parser.add_argument('-j', '--tgt_jd', required=False, default=None, metavar='J', \
                        type=int, help='JD day for fitting across JDs - only if \
                        deg_dim = "jd". Default to pick consecutive JD day')
    parser.add_argument('-r', '--rel_dir', required=False, default=None, metavar='R', \
                        type=str, help='Directory in which relative calibration \
                        results dataframes are located')
    parser.add_argument('-u', '--out_dir', required=False, default=None, metavar='U', \
                        type=str, help='Out directory to store dataframe')
    parser.add_argument('-n', '--new_df', required=False, action='store_true', \
                        help='Write data to a new csv file')
    parser.add_argument('-k', '--compression', required=False, default=None, metavar='K', \
                        type=str, help='Compression to use when pickling results dataframe')
    args = parser.parse_args()

    startTime = datetime.datetime.now()

    pjd = ''
    if args.deg_dim == 'jd':
        tgt_jd = args.tgt_jd
        if tgt_jd is None:
            # choose consecutive JD as default
            tgt_jd = int(float(args.jd_time)) + 1
        pjd = '.' + str(tgt_jd)

    out_fn = args.out
    if out_fn is None:
        out_fn = 'deg_df.{}.{}.{}{}.{}'.format(args.jd_time, args.pol, \
                                               args.deg_dim, pjd, args.dist)
    if args.out_dir is not None:
        if not os.path.exists(args.out_dir):
            os.mkdir(args.out_dir)
        out_fn = os.path.join(args.out_dir, out_fn)

    out_csv = fn_format(out_fn, 'csv')
    out_pkl = out_csv.rsplit('.', 1)[0] + '.pkl'
    csv_exists = os.path.exists(out_csv)
    pkl_exists = os.path.exists(out_pkl)
    if csv_exists or pkl_exists:
        if args.new_df:
            out_csv = new_fn(out_csv, None, startTime)
            out_pkl = out_csv.rsplit('.', 1)[0] + '.pkl'
            csv_exists = False
            pkl_exists = False

    freq_chans = mod_str_arg(args.chans)
    time_ints = mod_str_arg(args.tints)

    rel_df_path = find_rel_df(args.jd_time, args.pol, args.dist, args.rel_dir)
    rel_df = pd.read_pickle(rel_df_path)

    # retrieving visibility metadata
    md_fn = 'rel_df.{}.{}.md.pkl'.format(args.jd_time, args.pol)
    if args.rel_dir is not None:
        md_fn = os.path.join(args.rel_dir, md_fn)
    with open(md_fn, 'rb') as f:
        md = pickle.load(f)
    ant_pos = md['antpos']
    no_unq_bls = md['no_unq_bls']
    RedG = md['redg']
    ant_sep = red_ant_sep(RedG, ant_pos)

    pchans = args.chans
    if pchans is None:
        pchans = '0~{}'.format(md['Nfreqs']-1)
    ptints = args.tints
    if ptints is None:
        ptints = '0~{}'.format(md['Ntimes']-1)
    pdict = {'freq':'frequency channels', 'tint':'time integrations', \
             'jd':'Julian days'}
    print('Running degenerate translation on adjacent {} for visibility dataset {} '\
          'for frequency channel(s) {} and time integration(s) {} with {} '\
          'assumed noise distribution\n'.format(pdict[args.deg_dim], \
          os.path.basename(find_zen_file(args.jd_time)), pchans, ptints, args.dist))

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

    if args.deg_dim == 'jd':
        indices = ['time_int1', 'time_int2', 'freq']
        # find dataset from specified JD that contains visibilities at the same LAST
        jd_time2 = match_lst(args.jd_time, tgt_jd)
        if len(str(jd_time2)) < 13:
            jd_time2 = str(jd_time2) + '0' # add a trailing 0 that is omitted in float
        rel_df_path2 = find_rel_df(jd_time2, args.pol, args.dist, args.rel_dir)
        if isinstance(jd_time2, str):
            jd_time2 = float(jd_time2)
        # aligning datasets in LAST
        last_df = pd.read_pickle('jd_lst_map_idr2.pkl')
        last1 = last_df[last_df['JD_time'] == float(args.jd_time)]['LASTs'].values[0]
        last2 = last_df[last_df['JD_time'] == jd_time2]['LASTs'].values[0]
        _, offset = find_nearest(last2, last1[0])

        rel_df2 = pd.read_pickle(rel_df_path2)
        rel_df2 = rel_df2[rel_df2.index.get_level_values('time_int') >= offset]

        next_row = numpy.where(last_df['JD_time'] == jd_time2)[0][0] + 1
        rel_df_path3 = find_rel_df(last_df.iloc[next_row]['JD_time'], args.pol, \
                                   args.dist, args.rel_dir)
        rel_df3 = pd.read_pickle(rel_df_path3)
        rel_df3 = rel_df3[rel_df3.index.get_level_values('time_int') < offset]

        # combined results dataframes that is now alinged in LAST by row number
        # with rel_df:
        rel_df_c = pd.concat([rel_df2, rel_df3])
        # pairing time_ints from rel_df and rel_df_c that match in LAST
        # time_ints2 = rel_df_c.index.get_level_values('time_int').unique().values
        time_ints2 = numpy.arange(offset, offset + md['Ntimes']) % md['Ntimes']
        iter_dims = [idim for idim in zip(time_ints, time_ints2[time_ints])]
        iter_dims = [idim+(freq_chan,) for idim in iter_dims for freq_chan in \
                     freq_chans]
        iter_dims = sorted(iter_dims, key=lambda row: row[2]) # iterate across
        # LAST first - should speed up fitting
        a, b, c, d = 2, 2, 0, 1 # for iteration indexing
    else:
        rel_df_c = rel_df

    if not iter_dims:
        raise ValueError('No frequency channels or time integrations to '\
                         'iterate over - check that the specified --chans '\
                         'and --tints exist in the relative calibration '\
                         'results dataframes')

    skip_cal = False
    # skipping freqs and tints that are already in dataframe
    if csv_exists or pkl_exists:
        if csv_exists:
            df = pd.read_csv(out_csv, usecols=indices)
            idx_arr = df.values
        elif pkl_exists:
            df_pkl = pd.read_pickle(out_pkl)
            idx_arr = df_pkl.reset_index()[indices].values
        iter_dims = [idim for idim in iter_dims if not \
            numpy.equal(idx_arr, numpy.asarray(idim)).all(1).any()]
        if not any(iter_dims):
            print('Solutions to all specified frequency channels and time '\
                  'integrations already exist in {}\n'.format(out_pkl))
            skip_cal = True

    if not skip_cal:
        # removing 'jac', 'hess_inv', 'nfev', 'njev'
        slct_keys = ['success', 'status', 'message', 'fun', 'nit', 'x']
        no_deg_params = 3 # overall amplitude, x phase gradient, y phase gradient
        header = slct_keys[:-1] + list(numpy.arange(no_deg_params)) + indices

        stdout = io.StringIO()
        with redirect_stdout(stdout): # suppress output
            with open(out_csv, 'a') as f: # write / append to csv file
                writer = DictWriter(f, fieldnames=header)
                if not csv_exists:
                    writer.writeheader()
                initp = None
                for iter_dim in iter_dims:
                    # get relatively calibrated solutions
                    resx1 = rel_df.loc[iter_dim[a], iter_dim[c]][len(slct_keys)-1:-2]\
                    .values.astype(float)
                    resx2 = rel_df_c.loc[iter_dim[b], iter_dim[d]][len(slct_keys)-1:-2]\
                    .values.astype(float)
                    rel_vis1, _ = split_rel_results(resx1, no_unq_bls, coords=args.coords)
                    rel_vis2, _ = split_rel_results(resx2, no_unq_bls, coords=args.coords)

                    res_deg = doDegVisVis(ant_sep, rel_vis1, rel_vis2, \
                                          distribution=args.dist, initp=initp)
                    res_deg = {key:res_deg[key] for key in slct_keys}
                    # expanding out the solution
                    for i, param in enumerate(res_deg['x']):
                        res_deg[i] = param
                    # to use solution for next solve in iteration
                    # if res_deg['success']:
                    #     initp = res_deg['x']
                    del res_deg['x']
                    res_deg.update({indices[i]:iter_dim[i] for i in \
                                    range(no_deg_params)})
                    writer.writerow(res_deg)

        print('Degenerate fitting results saved to csv file {}'.format(out_csv))
        df = pd.read_csv(out_csv)
        df_indices = indices.copy()
        mv_col = df_indices.pop(1)
        df.set_index(df_indices, inplace=True)
        cols = list(df.columns.values)
        cols.remove(mv_col)
        df = df[[mv_col]+cols]
        # we now append the residuals as additional columns
        df = append_residuals_deg(df, rel_df, rel_df_c, md, out_fn=None)
        if pkl_exists and not csv_exists:
            df = pd.concat([df, df_pkl])
        df.sort_values(by=indices, inplace=True)
        if args.compression is not None:
            out_pkl += '.{}'.format(args.compression)
            print('{} compression used in pickling the dataframe'.format(args.compression))
        df.to_pickle(out_pkl, compression=args.compression)
        print('Degenerate fitting results dataframe pickled to {}'.format(out_pkl))

    print('Script run time: {}'.format(datetime.datetime.now() - startTime))


if __name__ == '__main__':
    main()
