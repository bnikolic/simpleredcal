"""Functions to assess the quality of calibration results"""


import numpy
import pandas as pd

from simpleredcal.red_likelihood import degVis, group_data, gVis, makeCArray, makeEArray, \
red_ant_sep, split_rel_results, XDgVis
from simpleredcal.red_utils import find_flag_file, find_zen_file, get_bad_ants


def norm_residuals(x_meas, x_pred):
    """Evaluates the residual between the measured and predicted quantities,
    normalized by the absolute value of their product

    :param x_meas: Measurement
    :type x_meas: ndarray
    :param x_pred: Prediction
    :type x_pred: ndarray

    :return: Normalized residual
    :rtype: ndarray
    """
    return (x_meas - x_pred) / numpy.sqrt(numpy.abs(x_meas)*numpy.abs(x_pred))


def abs_residuals(residuals):
    """Return median absolute residuals for both real and imag parts

    :param residuals: Residuals
    :type residuals: ndarray

    :return: Median absolute residuals, separately for Re and Im
    :rtype: list
    """
    return [numpy.median(numpy.absolute(getattr(residuals, i))) \
            for i in ('real', 'imag')]


def append_residuals_rel(rel_df, cdata, credg, coords, out_fn=None):
    """Calculates the residuals and normalized residuals for the relative redundant
    calibration fitting, for each frequency and time integration slice, and
    appends the residual results to the existing relative calibration results
    dataframe.

    :param rel_df: Relative calibration results dataframe
    :type rel_df: DataFrame
    :param cdata: Grouped visibilities with flags in numpy MaskedArray format,
    with format consistent with redg and dimensions (freq chans,
    time integrations, baselines)
    :type cdata: MaskedArray
    :param credg: Grouped baselines, condensed so that antennas are
    consecutively labelled. See relabelAnts
    :type credg: ndarray
    :param coords: Coordinate system in which gain and visibility parameters
    have been set up
    :type coords: str {"cartesian", "polar"}
    :param out_fn: Output dataframe file name. If None, file not pickled.
    :type out_fn: str, None

    :return: Relative calibration results dataframe, with residual columns appended
    :rtype: DataFrame
    """
    residual_cols = ['residual', 'norm_residual']
    if set(residual_cols).issubset(rel_df.columns.values):
        print('Residuals already appended to dataframe - exiting')
    else:
        print('Appending residuals to dataframe')
        no_unq_bls = credg[:, 0].max() + 1
        idxs = list(rel_df.index.names)
        rel_df.reset_index(inplace=True)
        freqs = rel_df['freq'].unique()
        tints = rel_df['time_int'].unique()
        if len(cdata.shape) == 4:
            # then dataset is over JDs
            xd = True
        else:
            xd = False

        def calc_rel_residuals(row):
            """Calculate residual and normalized residual to append to relative
            calibration results dataframe

            :param row: Row of the relative calibration results dataframe
            :type row: Series

            :return: Residual columns to append to dataframe
            :rtype: Series
            """
            cidx = len([col for col in rel_df.columns.values if not col.isdigit()])
            cmap_f = dict(map(reversed, enumerate(freqs)))
            cmap_t = dict(map(reversed, enumerate(tints)))
            resx = row.values[cidx:].astype(float)
            res_rel_vis, res_rel_gains = split_rel_results(resx, no_unq_bls, \
                                                           coords)
            if xd:
                gvisc = XDgVis
                res_rel_gains = res_rel_gains.reshape((cdata.shape[0], -1))
                res_rel_vis = numpy.tile(res_rel_vis, cdata.shape[0]).\
                                reshape((cdata.shape[0], -1))
                obs_vis = cdata[:, cmap_f[row['freq']], cmap_t[row['time_int']], :]
            else:
                gvisc = gVis
                obs_vis = cdata[cmap_f[row['freq']], cmap_t[row['time_int']], :]
            pred_rel_vis = gvisc(res_rel_vis, credg, res_rel_gains)
            rel_residuals = obs_vis - pred_rel_vis
            norm_rel_residuals = norm_residuals(obs_vis, pred_rel_vis)
            return pd.Series([rel_residuals, norm_rel_residuals])

        rel_df[residual_cols] = rel_df.apply(lambda row: calc_rel_residuals(row), \
                                             axis=1)
        rel_df.set_index(idxs, inplace=True)

        if out_fn is not None:
            rel_df.to_pickle(out_fn)

    return rel_df


def append_residuals_deg(deg_df, rel_df1, rel_df2, md, out_fn=None):
    """Calculates the residuals and normalized residuals for the degenerate
    parameter fitting for each frequency and time integration slice, and
    appends the residual results to the existing degenerate fitting results
    dataframe.

    :param deg_df: Degenerate fitting results dataframe
    :type deg_df: DataFrame
    :param rel_df1:
    :type rel_df1: DataFrame
    :param rel_df2:
    :type rel_df2: DataFrame
    :param md: Metadata for visibility dataset corresponding to deg_df
    :type md: dict
    :param out_fn: Output dataframe file name. If None, file not pickled.
    :type out_fn: str, None

    :return: Degenerate fitting results dataframe, with residual columns appended
    :rtype: DataFrame
    """
    residual_cols = ['residual', 'norm_residual']
    if set(residual_cols).issubset(deg_df.columns.values):
        print('Residuals already appended to dataframe - exiting')
    else:
        print('Appending residuals to dataframe')
        ant_sep = red_ant_sep(md['redg'], md['antpos'])
        idxs = list(deg_df.index.names)
        deg_df.reset_index(inplace=True)
        deg_cols = deg_df.columns.values

        def calc_deg_residuals(row):
            """Calculate residual and normalized residual to append to degenerate
            fitting results dataframe

            :param row: Row of the degenerate fitting results dataframe
            :type row: Series

            :return: Residual columns to append to dataframe
            :rtype: Series
            """
            cidx = len([col for col in rel_df1.columns.values if not \
                   col.isdigit()]) - 2
            if 'time_int' not in deg_cols:
                A = row['freq']
                B = A
                C = row['time_int1']
                D = row['time_int2']
            elif 'freq' not in deg_cols:
                A = row['freq1']
                B = row['freq2']
                C = row['time_int']
                D = C
            resx1 = rel_df1.loc[A, C][cidx:-2]\
            .values.astype(float)
            resx2 = rel_df2.loc[B, D][cidx:-2]\
            .values.astype(float)
            rel_vis1, _ = split_rel_results(resx1, md['no_unq_bls'])
            rel_vis2, _ = split_rel_results(resx2, md['no_unq_bls'])
            deg_w_alpha = degVis(ant_sep, rel_vis1, *row[-3:].values.astype(float))
            deg_residuals = rel_vis2 - deg_w_alpha
            norm_deg_residuals = norm_residuals(rel_vis2, deg_w_alpha)
            return pd.Series([deg_residuals, norm_deg_residuals])

        deg_df[residual_cols] = deg_df.apply(lambda row: calc_deg_residuals(row), \
                                             axis=1)
        deg_df.set_index(idxs, inplace=True)

        if out_fn is not None:
            deg_df.to_pickle(out_fn)

    return deg_df


def append_residuals_opt(opt_df, cdata, credg, out_fn=None):
    """Calculates the residuals and normalized residuals for the absolute
    optimal calibration fitting, for each frequency and time integration slice,
    and appends the residual results to the existing relative calibration results
    dataframe.

    :param opt_df: Absolute optimal calibration results dataframe
    :type opt_df: DataFrame
    :param cdata: Grouped visibilities with flags in numpy MaskedArray format,
    with format consistent with redg and dimensions (freq chans,
    time integrations, baselines)
    :type cdata: MaskedArray
    :param credg: Grouped baselines, condensed so that antennas are
    consecutively labelled. See relabelAnts
    :type credg: ndarray
    :param out_fn: Output dataframe file name. If None, file not pickled.
    :type out_fn: str, None

    :return: Absolute optimal calibration results dataframe, with residual
    columns appended
    :rtype: DataFrame
    """
    residual_cols = ['residual', 'norm_residual']
    if set(residual_cols).issubset(opt_df.columns.values):
        print('Residuals already appended to dataframe - exiting')
    else:
        print('Appending residuals to dataframe')
        no_ants = credg[:, 1:].max() + 1
        no_unq_bls = credg[:, 0].max() + 1
        idxs = list(opt_df.index.names)
        opt_df.reset_index(inplace=True)
        freqs = opt_df['freq'].unique()
        tints = opt_df['time_int'].unique()

        def calc_opt_residuals(row):
            """Calculate residual and normalized residual to append to absolute
            optimal calibration results dataframe

            :param row: Row of the relative calibration results dataframe
            :type row: Series

            :return: Residual columns to append to dataframe
            :rtype: Series
            """
            cidx = len([col for col in opt_df.columns.values if not col.isdigit()])
            cmap_f = dict(map(reversed, enumerate(freqs)))
            cmap_t = dict(map(reversed, enumerate(tints)))
            opt_resx = row.values[cidx:].astype(float)
            new_gain_comps = opt_resx[:2*no_ants]
            new_gains = makeEArray(new_gain_comps)
            w_alpha_comps = opt_resx[-no_unq_bls*2:]
            w_alpha = makeCArray(w_alpha_comps)
            obs_vis = cdata[cmap_f[row['freq']], cmap_t[row['time_int']], :]
            pred_opt_vis = gVis(w_alpha, credg, new_gains)
            opt_residuals = obs_vis - pred_opt_vis
            norm_opt_residuals = norm_residuals(obs_vis, pred_opt_vis)
            return pd.Series([opt_residuals, norm_opt_residuals])

        opt_df[residual_cols] = opt_df.apply(lambda row: calc_opt_residuals(row), \
                                             axis=1)
        opt_df.set_index(idxs, inplace=True)

        if out_fn is not None:
            opt_df.to_pickle(out_fn)

    return opt_df
