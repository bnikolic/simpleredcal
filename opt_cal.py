"""Batch absolute optimal calibration of relatively calibrated visibilities

The degenerate parameters arising in redundant relativecalibration are
constrained such that the average amplitude of antenna gains is set to 1, the
average phase of antenna gains is set to 0, the overall phase if set to 0 and
the phase gradients are 0.

example run:
$ python opt_cal.py 2458098.43869 --pol ee --chans 300~301 --tints 10~11

Can then read the results dataframe with:
> pd.read_pickle('opt_df.2458098.43869.ee.cauchy.pkl')

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

from red_likelihood import decomposeCArray, degVis, doOptCal, group_data, \
red_ant_sep
from red_utils import find_nearest, find_flag_file, find_rel_df, find_zen_file, \
fn_format, get_bad_ants, mod_str_arg, new_fn, split_rel_results


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.\
    RawDescriptionHelpFormatter, description=textwrap.dedent("""
    Absolute optimal calibration of relatively calibrated visibility solutions

    Takes the relatively calibrated visibility solutions of HERA data, and
    constrains their degenerate parameters, such that average amplitude of
    antenna gains is set to 1, the average phase of antenna gains is set to 0,
    the overall phase if set to 0 and the phase gradients are 0.

    Returns a pickled pandas dataframe of the Scipy optimization results for
    the absolute optimal calibration.
    """))
    parser.add_argument('jd_time', help='Fractional JD time of dataset to \
                        analyze', metavar='JD', type=float)
    parser.add_argument('-o', '--out', required=False, default=None, \
                        metavar='O', type=str, help='Output csv and df name')
    parser.add_argument('-p', '--pol', required=True, metavar='P', type=str, \
                        help='Polarization {"ee", "en", "nn", "ne"}')
    parser.add_argument('-c', '--chans', required=False, default=None, metavar='C', \
                        type=str, help='Frequency channels to fit {0, 1023}')
    parser.add_argument('-t', '--tints', required=False, default=None, metavar='T', \
                        type=str, help='Time integrations to fit {0, 59}')
    parser.add_argument('-m', '--dist', required=False, default='cauchy', metavar='F', \
                        type=str, help='Fitting distribution for calibration \
                        {"cauchy", "gaussian"}')
    parser.add_argument('-a', '--ref_ant', required=False, default=85, metavar='A', \
                        type=int, help='Reference antenna to set the overall \
                        phase')
    parser.add_argument('-r', '--rel_dir', required=False, default=None, metavar='R', \
                        type=str, help='Directory in which relative calibration \
                        results dataframes are located')
    parser.add_argument('-n', '--new_df', required=False, action='store_true', \
                        help='Write data to a new csv file')
    args = parser.parse_args()

    startTime = datetime.datetime.now()

    out_fn = args.out
    if out_fn is None:
        out_fn = 'opt_df.{}.{}.{}'.format(args.jd_time, args.pol, args.dist)

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

    zen_fn = find_zen_file(args.jd_time)
    bad_ants = get_bad_ants(zen_fn)
    flag_fn = find_flag_file(args.jd_time, 'first') # returns None if not found

    rel_df_path = find_rel_df(args.jd_time, args.pol, args.dist, args.rel_dir)
    rel_df = pd.read_pickle(rel_df_path)

    # retrieving visibility metadata
    md_fn = 'rel_df.{}.{}.md.pkl'.format(args.jd_time, args.pol)
    with open(md_fn, 'rb') as f:
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
    print('Running absolute optimal calibration for visibility dataset {} '\
          'for frequency channel(s) {} and time integration(s) {}\n'.\
          format(os.path.basename(zen_fn), pchans, ptints))

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

    indices = ['freq', 'time_int']
    no_tints = len(time_ints)
    iter_dims = list(numpy.ndindex((len(freq_chans), no_tints)))

    if not iter_dims:
        raise ValueError('No frequency channels or time integrations to '\
            'iterate over - check that the specified --chans and --tints exist '\
            'in the relative calibration results dataframes')

    skip_cal = False
    # skipping freqs and tints that are already in dataframe
    if csv_exists or pkl_exists:
        cmap_f = dict(map(reversed, enumerate(freq_chans)))
        cmap_t = dict(map(reversed, enumerate(time_ints)))
        if csv_exists:
            df = pd.read_csv(out_csv, usecols=indices)
            idx_arr = df.values
        elif pkl_exists:
            df = pd.read_pickle(out_pkl)
            idx_arr = df.reset_index()[indices].values
        done = [(cmap_f[f], cmap_t[t]) for (f, t) in idx_arr if (f in freq_chans \
        and t in time_ints)]
        iter_dims = [idim for idim in iter_dims if idim not in done]
        if not any(iter_dims):
            print('Solutions to all specified frequency channels and time '\
                  'integrations already exist in {}\n'.format(out_pkl))
            skip_cal = True

    if not skip_cal:
        hd, RedG, cData = group_data(zen_fn, args.pol, freq_chans, time_ints, \
                                     bad_ants, flag_path=flag_fn)
        flags = cData.mask
        cData = cData.data

        no_ants = numpy.unique(RedG[:, 1:]).size
        no_unq_bls = numpy.unique(RedG[:, 0]).size

        # removing 'jac', 'hess_inv', 'nfev', 'njev'
        slct_keys = ['success', 'status', 'message', 'fun', 'nit', 'x']
        no_deg_params = 4 # overall amplitude, overall phase, x & y phase gradients
        psize = no_ants*2 + no_deg_params + no_unq_bls*2
        header = slct_keys[:-1] + list(numpy.arange(psize)) + indices

        ant_sep = red_ant_sep(RedG, hd.antpos)
        def get_w_alpha(res_rel_vis, new_deg_params):
            """Apply degenerate parameters found from optimal absolute
            calibration to visibility solutions from relative redundant
            calibration

            :param res_rel_vis: Visibility solutions
            :type res_rel_vis: ndarray
            :param new_deg_params: Degenerate parameters
            optimal calibration
            :type new_deg_params: ndarray

            :return: Degenerately transformed visibility solutions
            :rtype: ndarray
            """
            return degVis(ant_sep, res_rel_vis, *new_deg_params[[0, 2, 3]])

        stdout = io.StringIO()
        with redirect_stdout(stdout): # suppress output
            with open(out_csv, 'a') as f: # write / append to csv file
                writer = DictWriter(f, fieldnames=header)
                if not csv_exists:
                    writer.writeheader()
                initp = None
                for i, iter_dim in enumerate(iter_dims):
                    # get absolute optimal calibrated solutions
                    rel_idim = (freq_chans[iter_dim[0]], time_ints[iter_dim[1]])
                    res_rel_vis, _ = split_rel_results(rel_df.loc[rel_idim]\
                        [len(slct_keys)-1:-2].values.astype(float), no_unq_bls)
                    res_opt = doOptCal(RedG, cData[iter_dim], hd.antpos, \
                        res_rel_vis, distribution=args.dist, ref_ant=args.ref_ant, \
                        initp=initp)
                    res_opt = {key:res_opt[key] for key in slct_keys}
                    # get the new visibility solutions
                    w_alpha = get_w_alpha(res_rel_vis, res_opt['x'][-no_deg_params:])
                    w_alpha_comps = decomposeCArray(w_alpha)
                    all_params = numpy.append(res_opt['x'], w_alpha_comps)
                    # expanding out the solution
                    for j, param in enumerate(all_params):
                        res_opt[j] = param
                    # to use solution for next solve in iteration
                    if res_opt['success']:
                        initp = res_opt['x']
                    # reset initp after each frequency slice
                    if not i%no_tints:
                        initp = None
                    del res_opt['x']
                    res_opt.update({indices[0]:rel_idim[0], \
                                    indices[1]:rel_idim[1]})
                    writer.writerow(res_opt)

        print('Absolute optimal calibration results saved to csv file {}'\
              .format(out_csv))
        df = pd.read_csv(out_csv)
        df.set_index(indices, inplace=True)
        df.sort_values(by=indices, inplace=True)
        df.to_pickle(out_pkl)
        print('Absolute optimal calibration results dataframe pickled to {}'\
              .format(out_pkl))

    print('Script run time: {}'.format(datetime.datetime.now() - startTime))


if __name__ == '__main__':
    main()
