"""Check the relative redundant calibration of solutions

example run:
$ python check_rel_cal.py 'rel_df.2458098.43869.ee.cauchy.pkl'
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

from simpleredcal.fit_diagnostics import norm_residuals
from simpleredcal.red_likelihood import doRelCal, group_data, norm_rel_sols, relabelAnts, \
split_rel_results
from simpleredcal.red_utils import find_zen_file, fn_format, get_bad_ants, new_fn


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.\
    RawDescriptionHelpFormatter, description=textwrap.dedent("""
    Check the relative redundant calibration of visibilities

    The relative calibration script rel_cal.py reuses the previous solution to
    initialize the next solver. While this greatly speeds up the code, there is
    the worry that the solver gets stuck in a local minimum. This script takes a
    random sample of frequency and time integration slices from a dataframe of
    results from relative calibration, and re-runs relative calibration on them
    with vanilla initial parameter guesses of 1+1j for the gains and 0+0j for
    the visibilities, and verifies if these results match with the original
    results by checking both their negative log-likelhoods and their gain
    amplitudes. We expect these to match, but we do not expect the gain and
    visibility solution phases to be equal, since there are still some additional
    degeneracies (overall phase & tilt shifts) that have not been accounted for.

    Returns a pickled pandas dataframe of the Scipy optimization results for
    the relative redundant calibration for each set of randomly chosen frequency
    channel and time integration slices chosen from the rel_df results dataframe
    """))
    parser.add_argument('rel_df', help='Relative calibration results dataframe \
                        in pickle file format', metavar='df', type=str)
    parser.add_argument('-o', '--out', required=False, default=None, \
                        metavar='O', type=str, help='Output csv and df name')
    parser.add_argument('-c', '--no_checks', required=False, default=50, \
                        metavar='C', type=int, help='Number of checks')
    parser.add_argument('-t', '--tol', required=False, default=0.01, \
                        metavar='T', type=float, help='Tolerance for the \
                        negative log-likelihood of relative calibration results \
                        to match')
    parser.add_argument('-w', '--overwrite', required=False, action='store_true', \
                        help='Overwrite existing check csv and dataframe')
    parser.add_argument('-r', '--rel_dir', required=False, default=None, metavar='R', \
                        type=str, help='Directory in which rel_dfs are stored')
    parser.add_argument('-u', '--out_dir', required=False, default=None, metavar='U', \
                        type=str, help='Out directory to store dataframe')
    parser.add_argument('-k', '--keep_csv', required=False, action='store_true', \
                        help='Keep csv file')
    args = parser.parse_args()

    startTime = datetime.datetime.now()

    sout = args.rel_df.split('.')
    jd_time = str('{}.{}'.format(sout[1], sout[2]))
    pol = sout[3]
    dist = sout[4]
    no_checks = args.no_checks

    out_fn = args.out
    if out_fn is None:
        out_fn = 'check_rel_df.{}.{}.{}'.format(jd_time, pol, dist)
    if args.out_dir is not None:
        if not os.path.exists(args.out_dir):
            os.mkdir(args.out_dir)
        out_fn = os.path.join(args.out_dir, out_fn)

    out_csv = fn_format(out_fn, 'csv')
    csv_exists = os.path.exists(out_csv)
    if csv_exists:
        if args.overwrite:
            os.remove(out_csv)

    out_df = out_csv.rsplit('.', 1)[0] + '.pkl'
    df_exists = os.path.exists(out_df)
    if df_exists:
        if args.overwrite:
            os.remove(out_df)
            df_exists = False

    match_keys = ['loglkl_match', 'gamp_match']
    if not df_exists:
        zen_fn = find_zen_file(jd_time)
        bad_ants = get_bad_ants(zen_fn)

        print('Checking the relative redundant calibration results for {}\n'.\
              format(args.rel_df))
        if args.rel_dir is not None:
            rel_dir_path = os.path.join(args.rel_dir, args.rel_df)
        else:
            rel_dir_path = args.rel_df
        rel_df = pd.read_pickle(rel_dir_path)
        no_checks = min(no_checks, len(rel_df.index))
        rnd_idxs = numpy.random.choice(rel_df.index.values, no_checks, \
                                       replace=False)
        rnd_chans = numpy.unique([rnd_idx[0] for rnd_idx in rnd_idxs])
        fmap = dict(map(reversed, enumerate(rnd_chans)))

        hd, RedG, cData = group_data(zen_fn, pol, rnd_chans, None, bad_ants)
        cData = cData.data

        freq_chans = numpy.arange(hd.Nfreqs)
        time_ints = numpy.arange(hd.Ntimes)

        # to get fields for the csv header
        no_ants = numpy.unique(RedG[:, 1:]).size
        no_unq_bls = numpy.unique(RedG[:, 0]).size
        cRedG = relabelAnts(RedG)
        psize = (no_ants + no_unq_bls)*2

        indices = ['freq', 'time_int']
        slct_keys = ['success', 'status', 'message', 'fun', 'nit', 'x']
        header = slct_keys[:-1] + match_keys + list(numpy.arange(psize)) + indices

        stdout = io.StringIO()
        with redirect_stdout(stdout): # suppress output
            with open(out_csv, 'a') as f: # write / append to csv file
                writer = DictWriter(f, fieldnames=header)
                writer.writeheader()
                for iter_dim in rnd_idxs:
                    res_rel = doRelCal(cRedG, cData[fmap[iter_dim[0]], iter_dim[1]], \
                                       no_unq_bls, no_ants, coords='cartesian', \
                                       distribution=dist, norm_gains=True)
                    res_rel = {key:res_rel[key] for key in slct_keys}

                    # checking results
                    res_rel[match_keys[0]] = numpy.abs(norm_residuals(rel_df.\
                        loc[iter_dim]['fun'], res_rel['fun'])) < args.tol
                    res_gamp = numpy.abs(split_rel_results(rel_df.loc[iter_dim][5:-2].\
                                         values.astype(float), no_unq_bls)[1])
                    check_gamp = numpy.abs(split_rel_results(res_rel['x'], \
                                           no_unq_bls)[1])
                    res_rel[match_keys[1]] = (numpy.abs(norm_residuals(res_gamp, \
                                              check_gamp)) < args.tol).all()

                    # expanding out the solution
                    for i, param in enumerate(res_rel['x']):
                        res_rel[i] = param
                    del res_rel['x']
                    res_rel.update({indices[0]:iter_dim[0], \
                                    indices[1]:iter_dim[1]})
                    writer.writerow(res_rel)

        df = pd.read_csv(out_csv)
        df.set_index(indices, inplace=True)
        df.sort_values(by=indices, inplace=True)
        df.to_pickle(out_df)

        if not args.keep_csv:
            os.remove(out_csv)
        else:
            print('Checked relative calibration results saved to csv file {}'.\
                  format(out_csv))

        print('Checked relative calibration results dataframe pickled to {}\n'.\
              format(out_df))

    else:
        df = pd.read_pickle(out_df)
        print('Checked relative calibration results already exists in {} - '\
              'specify --overwrite as an argument to perform check again.\n'.\
              format(out_df))
        no_checks = len(df.index)

    matches = df[match_keys].values
    all_match = matches.all()
    if all_match:
        pmatch = 'All'
    else:
        pmatch = '{}% of'.format(round(100*numpy.sum(matches[:, 1])/matches.shape[0], 2))
    print('{} iterations from the {} randomly selected frequency and time '\
          'slices match the original results at a tolerance of {}%.\n'.\
          format(pmatch, no_checks, args.tol*100))
    if not all_match:
        print('Mismatched iterations are {}\n'.format(df[~df['gamp_match']].index.values))

    print('Script run time: {}'.format(datetime.datetime.now() - startTime))

if __name__ == '__main__':
    main()
