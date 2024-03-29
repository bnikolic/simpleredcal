"""Batch relative redundant calibration of visibilities across frequencies
and time

*DEVELOPMENT IMPLEMENTATION*

example run:
$ python rel_cal.py '2458098.43869' --pol 'ee' --chans 300~301 --tints 0~1 \
--flag_type 'first' --dist 'cauchy' --method 'cartesian'

Can then read the dataframe with:
> pd.read_pickle('rel_df.2458098.43869.ee.cauchy.pkl')

Note that default is to write all solutions to the same csv file, for each
visibility dataset
"""


import argparse
import datetime
import functools
import io
import os
import pickle
import sys
import textwrap
from contextlib import redirect_stdout
from csv import DictWriter

import pandas as pd
import numpy

from hera_cal.io import HERAData

from simpleredcal.fit_diagnostics import append_residuals_rel
from simpleredcal.red_likelihood import doRelCal, doRelCalRP, flt_ant_pos, group_data, \
relabelAnts
from simpleredcal.red_utils import find_flag_file, find_nearest, find_rel_df, find_zen_file, \
fn_format, get_bad_ants, JD2LSTPATH, match_lst, mod_str_arg, new_fn


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.\
    RawDescriptionHelpFormatter, description=textwrap.dedent("""
    Relative redundant calibration of visibilities

    Takes a given HERA visibility dataset in uvh5 file format and performs
    relative redundant calibration (up to the overall amplitude, overall
    phase, and phase gradient degenerate parameters) for each frequency channel
    and each time integration in the dataset.

    Returns a pickled pandas dataframe of the Scipy optimization results for
    the relative redundant calibration for each set of frequency channel and
    time integration.
    """))
    parser.add_argument('jd_time', help='Fractional JD time of dataset to \
                        calibrate', metavar='JD', type=str)
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
    parser.add_argument('-f', '--flag_type', required=False, default='first', \
                        metavar='F', type=str, help='Flag type e.g. "first", \
                        "omni", "abs"')
    parser.add_argument('-d', '--dist', required=True, metavar='D', \
                        type=str, help='Fitting distribution for calibration \
                        {"cauchy", "gaussian"}')
    parser.add_argument('-m', '--method', required=False, default='cartesian', \
                        metavar='M', type=str, help='Method to use - {"cartesian", \
                        "polar", "RP"}, where RP stands for reduced parameters')
    parser.add_argument('-l', '--logamp', required=False, action='store_true', \
                        help='Use logamp method to force positive gain amplitudes')
    parser.add_argument('-g', '--tilt_reg', required=False, action='store_true', \
                        help='Add regularization term to constrain tilt shifts to 0')
    parser.add_argument('-a', '--gphase_reg', required=False, action='store_true', \
                        help='Add regularization term to constrain the gain phase mean')
    parser.add_argument('-i', '--initp_jd', required=False, default=None, metavar='I', \
                        type=int, help='JD of to find datasets to reuse initial parameters')
    parser.add_argument('-v', '--noise', required=False, action='store_true', \
                        help='Use noise from autos in nlogL calculations')
    parser.add_argument('-u', '--out_dir', required=False, default=None, metavar='U', \
                        type=str, help='Out directory to store dataframe')
    parser.add_argument('-n', '--new_df', required=False, action='store_true', \
                        help='Write data to a new dataframe')
    args = parser.parse_args()

    startTime = datetime.datetime.now()

    out_fn = args.out
    default_fn = 'rel_df.{}.{}.{}'.format(args.jd_time, args.pol, args.dist)
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

    zen_fn = find_zen_file(args.jd_time)
    bad_ants = get_bad_ants(zen_fn)

    flag_type = args.flag_type
    if flag_type is not None:
        flag_fn = find_flag_file(args.jd_time, flag_type)
    else:
        flag_fn = None

    freq_chans = mod_str_arg(args.chans)
    time_ints = mod_str_arg(args.tints)

    hd = HERAData(zen_fn)

    pchans = args.chans
    if pchans is None:
        pchans = '0~{}'.format(hd.Nfreqs-1)
    ptints = args.tints
    if ptints is None:
        ptints = '0~{}'.format(hd.Ntimes-1)
    print('Running relative redundant calibration on visibility dataset {} for '\
          'polarization {}, frequency channel(s) {} and time integration(s) {} '\
          'with {} assumed noise distribution\n'.\
          format(os.path.basename(zen_fn), args.pol, pchans, ptints, args.dist))

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
        grp = group_data(zen_fn, args.pol, freq_chans, time_ints, \
                         bad_ants, flag_path=flag_fn, noise=args.noise)
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
        psize = (no_ants + no_unq_bls)*2
        if args.tilt_reg:
            ant_pos_arr = flt_ant_pos(hd.antpos, ants)
        else:
            ant_pos_arr = None

        # discarding 'jac', 'hess_inv', 'nfev', 'njev'
        slct_keys = ['success', 'status', 'message', 'fun', 'nit', 'x']
        header = slct_keys[:-1] + list(numpy.arange(psize)) + indices

        # remove flagged channels from iter_dims
        if True in flags:
            flg_chans = numpy.where(flags.all(axis=(1,2)))[0] # indices
            print('Flagged channels for visibility dataset {} are: {}\n'.\
                 format(os.path.basename(zen_fn), freq_chans[flg_chans]))
            iter_dims = [idim for idim in iter_dims if idim[0] not in flg_chans]
            if not iter_dims: # check if slices to solve are empty
                print('All specified channels are flagged. Exiting.')
                sys.exit()


        if args.initp_jd is not None:
            jd_time2 = match_lst(args.jd_time, args.initp_jd)
            if len(str(jd_time2)) < 13:
                jd_time2 = str(jd_time2) + '0' # add a trailing 0 that is omitted in float
            rel_df_path1 = find_rel_df(jd_time2, args.pol, args.dist)
            if isinstance(jd_time2, str):
                jd_time2 = float(jd_time2)

            last_df = pd.read_pickle(JD2LSTPATH)
            last1 = last_df[last_df['JD_time'] == float(args.jd_time)]['LASTs'].values[0]
            last2 = last_df[last_df['JD_time'] == jd_time2]['LASTs'].values[0]
            _, offset = find_nearest(last2, last1[0])

            rel_df1 = pd.read_pickle(rel_df_path1)
            rel_df1 = rel_df1[rel_df1.index.get_level_values('time_int') >= offset]

            next_row = numpy.where(last_df['JD_time'] == jd_time2)[0][0] + 1
            rel_df_path2 = find_rel_df(last_df.iloc[next_row]['JD_time'], args.pol, \
                                       args.dist)
            rel_df2 = pd.read_pickle(rel_df_path2)
            rel_df2 = rel_df2[rel_df2.index.get_level_values('time_int') < offset]

            rel_df_c = pd.concat([rel_df1, rel_df2])

            # filter by specified channels and time integrations
            time_ints_offset = (time_ints + offset) % hd.Ntimes
            freq_flt = numpy.in1d(rel_df_c.index.get_level_values('freq'), freq_chans)
            tint_flt = numpy.in1d(rel_df_c.index.get_level_values('time_int'), time_ints_offset)
            rel_df_c = rel_df_c[freq_flt & tint_flt]

            time_ints2 = numpy.tile(rel_df_c.index.get_level_values('time_int').unique().values, freq_chans.size)
            iter_dims = [idim+(tint,) for idim, tint in zip(iter_dims, time_ints2)]

            phase_reg_initp = True
        else:
            phase_reg_initp = False


        def cal(credg, distribution, coords, no_unq_bls, no_ants, logamp, \
                tilt_reg, gphase_reg, ant_pos_arr, obsvis, noise, initp):
            """Relative redundant calibration with doRelCal: unconstrained
            minimizer using cartesian coordinates - this is the fastest solver

            :param credg: Grouped baselines, condensed so that antennas are
            consecutively labelled. See relabelAnts
            :type credg: ndarray
            :param distribution: Distribution to fit likelihood {'gaussian', 'cauchy'}
            :type distribution: str
            :param coords: Coordinate system in which gain and visibility parameters
            have been set up
            :type coords: str {"cartesian", "polar"}
            :param no_unq_bls: Number of unique baselines (equivalently the number of
            redundant visibilities)
            :type no_unq_bls: int
            :param no_ants: Number of antennas for given observation
            :type no_ants: int
            :param logamp: The logarithm of the amplitude initial parameters is taken,
            such that only positive solutions can be returned. Only if coords=="polar".
            :type logamp: bool
            :param tilt_reg: Add regularization term to constrain tilt shifts to 0
            :type tilt_reg: bool
            :param gphase_reg: Add regularization term to constrain the gain phase mean
            :type gphase_reg: bool
            :param ant_pos_arr: Array of filtered antenna position coordinates for the antennas
            in ants. See flt_ant_pos.
            :type ant_pos_arr: ndarray
            :param obsvis: Observed sky visibilities for a given frequency and given time,
            reformatted to have format consistent with redg
            :type obsvis: ndarray
            :param noise: Noise array to feed into log-likelihood calculations
            :type noise: ndarray
            :param initp: Initial parameter guesses for true visibilities and gains
            :type initp: ndarray, None

            :return: Optimization result for the solved antenna gains and true sky
            visibilities
            :rtype: Scipy optimization result object
            """
            res_rel, initp_new = doRelCal(credg, obsvis, no_unq_bls, no_ants, \
                coords=coords, distribution=distribution, noise=noise, \
                norm_gains=True, logamp=logamp, tilt_reg=tilt_reg, \
                gphase_reg=gphase_reg, ant_pos_arr=ant_pos_arr, initp=initp, \
                return_initp=True, phase_reg_initp=phase_reg_initp)
            res_rel = {key:res_rel[key] for key in slct_keys}
            # use solution for next solve in iteration
            if res_rel['success']:
                initp = initp_new
            return res_rel, initp

        def cal_RP(credg, distribution, no_unq_bls, no_ants, logamp, \
                   tilt_reg, gphase_reg, ant_pos_arr, obsvis, noise, initp):
            """Relative redundant calibration with doRelCalRP: constrained
            minimizer (by reducing the number of parameters) using polar
            coordinates

            :param credg: Grouped baselines, condensed so that antennas are
            consecutively labelled. See relabelAnts
            :type credg: ndarray
            :param distribution: Distribution to fit likelihood {'gaussian', 'cauchy'}
            :type distribution: str
            :param no_unq_bls: Number of unique baselines (equivalently the number of
            redundant visibilities)
            :type no_unq_bls: int
            :param no_ants: Number of antennas for given observation
            :type no_ants: int
            :param logamp: The logarithm of the amplitude initial parameters is taken,
            such that only positive solutions can be returned. Only if coords=="polar".
            :type logamp: bool
            :param tilt_reg: Add regularization term to constrain tilt shifts to 0
            :type tilt_reg: bool
            :param gphase_reg: Add regularization term to constrain the gain phase mean
            :type gphase_reg: bool
            :param ant_pos_arr: Array of filtered antenna position coordinates for the antennas
            in ants. See flt_ant_pos.
            :type ant_pos_arr: ndarray
            :param obsvis: Observed sky visibilities for a given frequency and given time,
            reformatted to have format consistent with redg
            :type obsvis: ndarray
            :param noise: Noise array to feed into log-likelihood calculations
            :type noise: ndarray
            :param initp: Initial parameter guesses for true visibilities and gains
            :type initp: ndarray, None

            :return: Optimization result for the solved antenna gains and true sky
            visibilities
            :rtype: Scipy optimization result object
            """
            res_rel, initp_ = doRelCalRP(credg, obsvis, no_unq_bls, no_ants, \
                distribution=distribution, noise=noise, constr_phase=True, \
                amp_constr='prod', bounded=True, logamp=logamp, tilt_reg=tilt_reg, \
                gphase_reg=gphase_reg, ant_pos_arr=gphase_reg, initp=initp)
            res_rel = {key:res_rel[key] for key in slct_keys}
            # use solution for next solve in iteration
            if res_rel['success']:
                initp = initp_
            return res_rel, initp

        if args.method.upper() == 'RP':
            RelCal = functools.partial(cal_RP, cRedG, args.dist, no_unq_bls, no_ants, \
                                       args.logamp, args.tilt_reg, args.gphase_reg, \
                                       ant_pos_arr)
            coords = 'polar'
        else:
            RelCal = functools.partial(cal, cRedG, args.dist, args.method, no_unq_bls, \
                                       no_ants, args.logamp, args.tilt_reg, \
                                       args.gphase_reg, ant_pos_arr)
            coords = args.method

        stdout = io.StringIO()
        with redirect_stdout(stdout): # suppress output
            with open(out_csv, 'a') as f: # write / append to csv file
                writer = DictWriter(f, fieldnames=header)
                if not csv_exists:
                    writer.writeheader()
                initp = None
                for i, iter_dim in enumerate(iter_dims):
                    if args.initp_jd is not None:
                        initp = rel_df_c.loc[(freq_chans[iter_dim[0]], iter_dim[2])]\
                                [len(slct_keys[:-1]):-2].values.astype(float)
                    if args.noise:
                        noisec = cNData[iter_dim[:2]]
                    res_rel, initp = RelCal(cData[iter_dim[:2]], noisec, initp)
                    # expanding out the solution
                    for j, param in enumerate(res_rel['x']):
                        res_rel[j] = param
                    # reset initp after each frequency slice
                    if not (i+1)%no_tints and args.initp_jd is None:
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
                _, _, cData = group_data(zen_fn, args.pol, freqs, tints, \
                                         bad_ants, flag_path=flag_fn)
                cData = cData.data
        df.set_index(indices, inplace=True)
        # we now append the residuals as additional columns
        df = append_residuals_rel(df, cData, cRedG, coords, out_fn=None)
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
                  'Ntimes':hd.Ntimes}
            with open(out_md, 'wb') as f:
                pickle.dump(md, f, protocol=pickle.HIGHEST_PROTOCOL)
            print('Relative calibration metadata pickled to {}\n'.format(out_md))

    print('Script run time: {}'.format(datetime.datetime.now() - startTime))


if __name__ == '__main__':
    main()
