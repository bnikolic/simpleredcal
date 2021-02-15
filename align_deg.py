"""Degenerately transform, align and merge redundantly calibrated
dataframes
"""

import argparse
import datetime
import os
import pickle
import textwrap

import pandas as pd
import numpy

from align_utils import align_df, idr2_jds
from red_likelihood import decomposeCArray, degVis, makeCArray, red_ant_sep
from red_utils import find_deg_df, find_nearest, find_rel_df, fn_format, \
match_lst, mod_str_arg


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.\
    RawDescriptionHelpFormatter, description=textwrap.dedent("""
    Degenerately transform, align and merge redundantly calibrated
    dataframes
    """))
    parser.add_argument('jd_time', help='Fractional JD time of dataframe to \
                        align other dataframes to', metavar='JD', type=str)
    parser.add_argument('-j', '--jd_comp', required=True, metavar='J', \
                        type=str, help='JDs of dataframes to align')
    parser.add_argument('-a', '--jd_anchor', required=True, metavar='A', \
                        type=int, help='JD of anchor day so that all other JDs \
                        are transformed to the degenerate space of this day')
    parser.add_argument('-p', '--pol', required=True, metavar='P', type=str, \
                        help='Polarization {"ee", "en", "nn", "ne"}')
    parser.add_argument('-d', '--dist', required=True, metavar='D', \
                        type=str, help='Noise distribution for calibration \
                        {"cauchy", "gaussian"}')
    parser.add_argument('-r', '--rel_dir', required=False, default='rel_dfs', metavar='R', \
                        type=str, help='Directory in which relative calibration \
                        results dataframes are located')
    parser.add_argument('-g', '--deg_dir', required=False, default='deg_dfs', metavar='G', \
                        type=str, help='Directory in which degenerate comparison \
                        results dataframes are located')
    parser.add_argument('-o', '--out', required=False, default=None, \
                        metavar='O', type=str, help='Output csv and df name')
    parser.add_argument('-u', '--out_dir', required=False, default=None, metavar='U', \
                        type=str, help='Out directory to store dataframe')
    args = parser.parse_args()

    startTime = datetime.datetime.now()

    out_fn = args.out
    if out_fn is None:
        out_fn = 'aligned_red_deg.{}.{}'.format(args.pol, args.dist)
    if args.out_dir is not None:
        if not os.path.exists(args.out_dir):
            os.mkdir(args.out_dir)
        out_fn = os.path.join(args.out_dir, out_fn)
    out_pkl = fn_format(out_fn, 'pkl')

    jd_comp = args.jd_comp
    if '_' in jd_comp:
        jd_comp = numpy.asarray(jd_comp.split('_'), dtype=int)
    else:
        jd_comp = mod_str_arg(args.jd_comp)
    jd_comp = numpy.intersect1d(jd_comp, idr2_jds)

    print('Finding relatively calibrated dataframes on JDs {} ({} polarization '\
          'and {} assumed noise during the calibration) and aligning them '\
          'in LAST to dataset {}, with all relatively calibrated datasets being '\
          'transformed to the degenerate space of {}.\n'.\
          format(numpy.append([int(float(args.jd_time))], jd_comp), args.pol, \
          args.dist, args.jd_time, args.jd_anchor))

    if os.path.exists(out_pkl):
        print('Overwriting {}.\n'.format(out_pkl))

    indices = ['freq', 'time_int']
    resid_cols = ['residual', 'norm_residual']
    min_list = ['success', 'status', 'message', 'fun', 'nit']

    with open(os.path.join(args.rel_dir, 'rel_df.{}.{}.md.pkl'.format(args.jd_time, \
              args.pol)), 'rb') as f:
        md = pickle.load(f)

    vis_list = list(map(str, numpy.arange(md['no_unq_bls']*2).tolist()))
    gain_list = list(map(str, numpy.arange(md['no_unq_bls']*2, (md['no_unq_bls'] + \
                     md['no_ants'])*2 ).tolist()))

    rel_df_path = find_rel_df(args.jd_time, args.pol, args.dist, args.rel_dir)
    rel_df = pd.read_pickle(rel_df_path)
    rel_df.drop(columns=resid_cols+gain_list, inplace=True)

    Nfreqs = rel_df.index.get_level_values('freq').unique().size
    Ntints = rel_df.index.get_level_values('time_int').unique().size

    if int(float(args.jd_time)) != args.jd_anchor:
        rel_df_d = rel_df[min_list].copy()
        rel_df_d = rel_df_d.reindex(columns=rel_df_d.columns.values.tolist() + vis_list)

        deg_df_path = find_deg_df(args.jd_time, args.pol, 'jd.{}'.format(args.jd_anchor), \
                                  args.dist, args.deg_dir)
        deg_df = pd.read_pickle(deg_df_path)
        deg_df_d = deg_df[['0', '1', '2']].copy().reset_index()
        deg_df_d.rename(columns={'time_int1': 'time_int', '0': 'amp', '1': 'tilt_x', \
                                 '2':'tilt_y'}, inplace=True)
        deg_df_d.set_index(indices, inplace=True)
        deg_df_d.sort_index(inplace=True)

        rel_df = rel_df.join(deg_df_d)

        ant_sep = red_ant_sep(md['redg'], md['antpos'])
        rel_df_d[vis_list] = rel_df.apply(lambda row: pd.Series(decomposeCArray(\
            degVis(ant_sep, makeCArray(row[len(min_list):len(min_list) + \
            md['no_unq_bls']*2].values.astype(float)), *row[-3:].values.astype(float)))), \
            axis=1)
    else:
        rel_df_d = rel_df

    new_indices = ['freq', 'time_int', 'JD']
    rel_df_d['JD'] = int(float(args.jd_time))
    rel_df_d.reset_index(inplace=True)
    rel_df_d.set_index(new_indices, inplace=True)
    rel_df_d.sort_index(inplace=True)

    for jd_ci in jd_comp:
        rel_dfk = align_df('rel', args.jd_time, jd_ci, args.rel_dir, args.dist, \
                           args.pol)
        rel_dfk.drop(columns=resid_cols+gain_list, inplace=True)

        if int(jd_ci) != args.jd_anchor:
            deg_dfk = align_df('deg', args.jd_time, jd_ci, args.deg_dir, args.dist, \
                               args.pol, JD_anchor=args.jd_anchor)

            # Degenerate transformation of redundant visibility solutions
            deg_dfk = deg_dfk[['0', '1', '2']].copy().reset_index()
            deg_dfk.rename(columns={'time_int1': 'time_int', '0': 'amp', \
                                    '1': 'tilt_x', '2':'tilt_y'}, inplace=True)
            deg_dfk.set_index(indices, inplace=True)
            deg_dfk.sort_index(inplace=True)
            rel_dfk = rel_dfk.join(deg_dfk)

            rel_df_di = rel_dfk[min_list].copy()
            rel_df_di = rel_df_di.reindex(columns=rel_df_di.columns.values.tolist() \
                                                  +vis_list)
            rel_df_di[vis_list] = rel_dfk.apply(lambda row: pd.Series(decomposeCArray(\
                degVis(ant_sep, makeCArray(row[len(min_list):len(min_list) + \
                md['no_unq_bls']*2].values.astype(float)), *row[-3:].values.astype(float)))), \
                axis=1)
            rel_dfk = rel_df_di

        rel_dfk['JD'] = jd_ci
        rel_dfk.reset_index(inplace=True)
        rel_dfk.set_index(new_indices, inplace=True)
        rel_dfk.sort_index(inplace=True)
        rel_df_d = pd.concat([rel_df_d, rel_dfk])

    rel_df_d.sort_index(inplace=True)
    rel_df_d.to_pickle(out_pkl)
    print('Aligned dataframe pickled to {}'.format(out_pkl))

    print('Script run time: {}'.format(datetime.datetime.now() - startTime))


if __name__ == '__main__':
    main()