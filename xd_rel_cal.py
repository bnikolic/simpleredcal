"""Batch across days relative redundant calibration of visibilities across frequencies
and time

example run:
$ python xd_rel_cal.py '2458098.43869' --jds 2458098~2458116 --pol 'ee' \
--chans 300~301 --tints 0~1 --flag_type 'first' --dist 'cauchy'

Can then read the dataframe with:
> pd.read_pickle('xd_rel_df.2458098.43869.ee.cauchy.pkl')

Note that default is to write all solutions to the same csv file, for each
visibility dataset
"""


import argparse
import datetime
import functools
import io
import os
import pickle
import textwrap
import warnings
from contextlib import redirect_stdout
from csv import DictWriter

import pandas as pd
import numpy

from hera_cal.io import HERAData

from align_utils import idr2_jds
from fit_diagnostics import append_residuals_rel
from red_likelihood import doRelCalD, relabelAnts
from red_utils import find_zen_file, fn_format, mod_str_arg, new_fn
from xd_utils import XDgroup_data

warnings.filterwarnings('ignore', \
    message='telescope_location is not set. Using known values for HERA.')
warnings.filterwarnings('ignore', \
    message='antenna_positions is not set. Using known values for HERA.')


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.\
    RawDescriptionHelpFormatter, description=textwrap.dedent("""
    Across days relative redundant calibration of visibilities

    Takes HERA visibility datasets across several JDs in uvh5 file format,
    aligns them in LAST and then performs relative redundant calibration
    (up to the overall amplitude, overall phase, and phase gradient degenerate
    parameters) for each frequency channel and each time integration in the dataset.

    Returns a pickled pandas dataframe of the Scipy optimization results for
    the relative redundant calibration for each set of frequency channel and
    time integration.
    """))
    parser.add_argument('jd_time', help='Fractional JD time of dataset to \
                        align other dataframes to', metavar='JD', type=str)
    parser.add_argument('-j', '--jds', required=True, metavar='J', \
                        type=str, help='JDs to calibrate')
    parser.add_argument('-p', '--pol', required=True, metavar='P', type=str, \
                        help='Polarization {"ee", "en", "nn", "ne"}')
    parser.add_argument('-c', '--chans', required=False, default=None, metavar='C', \
                        type=str, help='Frequency channels to calibrate \
                        {0, 1023}')
    parser.add_argument('-t', '--tints', required=False, default=None, metavar='T', \
                        type=str, help='Time integrations to calibrate \
                        {0, 59}')
    parser.add_argument('-f', '--flag_type', required=False, default='first', \
                        metavar='F', type=str, help='Flag type e.g. "first", \
                        "omni", "abs"')
    parser.add_argument('-d', '--dist', required=True, metavar='D', \
                        type=str, help='Fitting distribution for calibration \
                        {"cauchy", "gaussian"}')
    parser.add_argument('-v', '--noise', required=False, action='store_true', \
                        help='Use noise from autos in nlogL calculations')
    parser.add_argument('-o', '--out', required=False, default=None, \
                        metavar='O', type=str, help='Output csv and df name')
    parser.add_argument('-u', '--out_dir', required=False, default=None, metavar='U', \
                        type=str, help='Out directory to store dataframe')
    parser.add_argument('-n', '--new_df', required=False, action='store_true', \
                        help='Write data to a new dataframe')
    args = parser.parse_args()

    startTime = datetime.datetime.now()

    zen_fn = find_zen_file(args.jd_time)
    hd = HERAData(zen_fn)
    last = hd.lsts[0]

    out_fn = args.out
    default_fn = 'xd_rel_df.{}.{}.{}'.format('{:.4f}'.format(hd.lsts[0]), \
                                             args.pol, args.dist)
    if out_fn is None:
        out_fn = default_fn
    if args.out_dir is not None:
        if not os.path.exists(args.out_dir):
            os.mkdir(args.out_dir)
        out_fn = os.path.join(args.out_dir, out_fn)
        if out_fn is not None:
            default_fn = os.path.join(args.out_dir, default_fn)

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

    if '_' in args.jds:
        JDs = numpy.asarray(args.jds.split('_'), dtype=int)
    else:
        JDs = mod_str_arg(args.jds)
    JDs = numpy.intersect1d(JDs, idr2_jds)

    freq_chans = mod_str_arg(args.chans)
    time_ints = mod_str_arg(args.tints)

    pchans = args.chans
    if pchans is None:
        pchans = '0~{}'.format(hd.Nfreqs-1)
    ptints = args.tints
    if ptints is None:
        ptints = '0~{}'.format(hd.Ntimes-1)
    print('Running relative redundant calibration on visibility dataset {} for '\
          'polarization {}, JDs {}, frequency channel(s) {} and time integration(s) {} '\
          'with {} assumed noise distribution\n'.\
          format(os.path.basename(zen_fn), args.pol, ' '.join(map(str, JDs)), pchans, ptints, \
                 args.dist))

    if freq_chans is None:
        freq_chans = numpy.arange(hd.Nfreqs)
    if time_ints is None:
        time_ints = numpy.arange(hd.Ntimes)

    indices = ['freq', 'time_int']

    no_tints = len(time_ints)
    iter_dims = list(numpy.ndindex((len(freq_chans), no_tints)))
    skip_cal = False
    # skipping freqs and tints that are already in the dataframe
    if csv_exists or pkl_exists:
        cmap_f = dict(map(reversed, enumerate(freq_chans)))
        cmap_t = dict(map(reversed, enumerate(time_ints)))
        if csv_exists:
            df = pd.read_csv(out_csv, usecols=indices)
            idx_arr = df.values
        elif pkl_exists:
            df_pkl = pd.read_pickle(out_pkl)
            idx_arr = df_pkl.index.values
        done = [(cmap_f[f], cmap_t[t]) for (f, t) in idx_arr if (f in freq_chans \
        and t in time_ints)]
        iter_dims = [idim for idim in iter_dims if idim not in done]
        if not any(iter_dims):
            print('Solutions to all specified frequency channels and time '\
                  'integrations already exist in {}\n'.format(out_pkl))
            skip_cal = True

    if not skip_cal:
        stdout = io.StringIO()
        with redirect_stdout(stdout): # suppress output
            grp = XDgroup_data(args.jd_time, JDs, args.pol, chans=freq_chans, \
                            tints=time_ints, use_flags=args.flag_type, \
                            noise=args.noise)
        if not args.noise:
            _, RedG, cData = grp
            noisec = None
        else:
            _, RedG, cData, cNData = grp

        flags = cData.mask
        cData = cData.data

        # to get fields for the csv header
        ants = numpy.unique(RedG[:, 1:])
        no_ants = ants.size
        no_unq_bls = numpy.unique(RedG[:, 0]).size
        cRedG = relabelAnts(RedG)
        psize = (no_ants*JDs.size + no_unq_bls)*2

        # discarding 'jac', 'hess_inv', 'nfev', 'njev'
        slct_keys = ['success', 'status', 'message', 'fun', 'nit', 'x']
        header = slct_keys[:-1] + list(numpy.arange(psize)) + indices

        # remove flagged channels from iter_dims
        if isinstance(flags, numpy.bool_):
            # If all flags are the same
            flags = [flags]
        if True in flags:
            flg_chans = numpy.where(flags.all(axis=(1, 2)))[0] # indices
            print('Flagged channels for visibility dataset {} are: {}\n'.\
                 format(os.path.basename(zen_fn), freq_chans[flg_chans]))
            iter_dims = [idim for idim in iter_dims if idim[0] not in flg_chans]

        def cal(credg, distribution, no_unq_bls, no_ants, obsvis, noise, initp):
            """Relative redundant calibration across days with doRelCalD:
            default implementation with unconstrained minimizer using cartesian
            coordinates
            """
            res_rel, initp_new = doRelCalD(credg, obsvis, no_unq_bls, no_ants, \
                distribution=distribution, noise=noise, initp=initp, \
                return_initp=True, xd=True)
            res_rel = {key:res_rel[key] for key in slct_keys}
            # use solution for next solve in iteration
            if res_rel['success']:
                initp = initp_new
            return res_rel, initp

        RelCal = functools.partial(cal, cRedG, args.dist, no_unq_bls, no_ants)

        with redirect_stdout(stdout): # suppress output
            with open(out_csv, 'a') as f: # write / append to csv file
                writer = DictWriter(f, fieldnames=header)
                if not csv_exists:
                    writer.writeheader()
                initp = None
                for i, iter_dim in enumerate(iter_dims):
                    if args.noise:
                        noisec = cNData[:, iter_dim[0], iter_dim[1], :]
                    res_rel, initp = RelCal(cData[:, iter_dim[0], iter_dim[1], :], \
                                            noisec, initp)
                    # expanding out the solution
                    for j, param in enumerate(res_rel['x']):
                        res_rel[j] = param
                    # reset initp after each frequency slice
                    if not (i+1)%no_tints:
                        initp = None
                    del res_rel['x']
                    res_rel.update({indices[0]:freq_chans[iter_dim[0]], \
                                    indices[1]:time_ints[iter_dim[1]]})
                    writer.writerow(res_rel)

        print('Relative calibration results saved to csv file {}'.format(out_csv))
        df = pd.read_csv(out_csv)
        if csv_exists:
            freqs = df['freq'].unique()
            tints = df['time_int'].unique()
            if cData.shape[0] != freqs.size or cData.shape[1] != tints.size:
                _, _, cData = XDgroup_data(args.jd_time, JDs, args.pol, chans=freqs,
                                           tints=tints, use_flags=args.flag_type, \
                                           noise=None)
                cData = cData.data
        df.set_index(indices, inplace=True)
        # # we now append the residuals as additional columns
        # df = append_residuals_rel(df, cData, cRedG, 'cartesian', out_fn=None)
        if pkl_exists and not csv_exists:
            df = pd.concat([df, df_pkl])
        df.sort_values(by=indices, inplace=True)
        df.to_pickle(out_pkl)
        print('Relative calibration results dataframe pickled to {}'.format(out_pkl))

        # creating metadata file
        out_md = default_fn.rsplit('.', 1)[0] + '.md.pkl'
        if not os.path.exists(out_md):
            md = {'no_ants':no_ants, 'no_unq_bls':no_unq_bls, 'redg':RedG, \
                  'antpos':hd.antpos, 'last':hd.lsts, 'Nfreqs':hd.Nfreqs, \
                  'Ntimes':hd.Ntimes, 'JDs':JDs}
            with open(out_md, 'wb') as f:
                pickle.dump(md, f, protocol=pickle.HIGHEST_PROTOCOL)
            print('Relative calibration metadata pickled to {}\n'.format(out_md))

    print('Script run time: {}'.format(datetime.datetime.now() - startTime))


if __name__ == '__main__':
    main()
