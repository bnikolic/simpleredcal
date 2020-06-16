"""Functions to assess the quality of calibration results"""


import numpy
import pandas as pd

from red_likelihood import group_data, gVis, relabelAnts
from red_utils import find_flag_file, find_zen_file, get_bad_ants, \
split_rel_results


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


def append_residuals_rel(rel_df, pkl_df=False, out_fn=None, cdata=None, \
                         redg=None):
    """Calculates the residuals and normalized for the relative redundant
    calibration fitting, for each frequency and time integration slice, and
    appends the residual results to the existing relative calibration results
    dataframe.

    :param rel_df: (Path of) relative calibration results dataframe. Path
    must be to pickled dataframe.
    file format
    :type rel_df: DataFrame, str
    :param pkl_df: If True, saves dataframe to pickle. If rel_df is a str, and
    out_fn is not specified, then overwrites rel_df, else must specify out_fn.
    :type pkl_df: bool
    :param out_fn: Output dataframe file name
    :type out_fn: str
    :param cdata: Grouped visibilities with flags in numpy MaskedArray format,
    with format consistent with redg and dimensions (freq chans,
    time integrations, baselines)
    :type cdata: MaskedArray
    :param redg: Grouped baselines, as returned by groupBls
    :type redg: ndarray

    :return: Relative calibration results dataframe, with residual columns appended
    :rtype: DataFrame
    """
    if isinstance(rel_df, pd.DataFrame) and pkl_df:
        if not isinstance(out_fn, str):
            raise ValueError('Specify out_fn to save pickled dataframe')
    if isinstance(rel_df, str):
        if pkl_df and out_fn is None:
            out_fn = rel_df
        rel_df = pd.read_pickle(rel_df)
    if not isinstance(rel_df, pd.DataFrame):
        raise ValueError('rel_df must be either a relative calibration results '\
            'dataframe, or a path to such a dataframe in pickle file format')

    residual_cols = ['residual', 'norm_residual']
    if set(residual_cols).issubset(rel_df.columns.values):
        print('Residuals already appended to dataframe - exiting')
    else:
        print('Appending residuals to dataframe')

        if cdata is None or redg is None:
            sout = out_fn.split('.')
            jd_time = float('{}.{}'.format(sout[1], sout[2]))
            pol = sout[3]
            dist = sout[4]

            zen_fn = find_zen_file(jd_time)
            bad_ants = get_bad_ants(zen_fn)
            flags_fn = find_flag_file(jd_time, 'first')

            hdraw, redg, cMData = group_data(zen_fn, pol, None, None, bad_ants, \
                                             flags_fn)
            cdata = cMData.filled()

        no_unq_bls = numpy.unique(redg[:, 0]).size

        idxs = list(rel_df.index.names)
        rel_df.reset_index(inplace=True)

        freqs = rel_df['freq'].unique()
        tints = rel_df['time_int'].unique()

        def calc_residuals(row):
            """Calculate residual and normalized residual to append to results
            dataframe

            :param row: Row of the relative calibration results dataframe
            :type row: Series

            :return: Residual columns to append to dataframe
            :rtype: Series
            """
            cidx = len([col for col in rel_df.columns.values if not col.isdigit()])
            cmap_f = dict(map(reversed, enumerate(freqs)))
            cmap_t = dict(map(reversed, enumerate(tints)))
            resx = row.values[cidx:].astype(float)
            res_rel_vis, res_rel_gains = split_rel_results(resx, no_unq_bls)
            obs_vis = cdata[cmap_f[row['freq']], cmap_t[row['time_int']], :]
            pred_rel_vis = gVis(res_rel_vis, relabelAnts(redg), res_rel_gains)
            rel_residuals = obs_vis - pred_rel_vis
            norm_rel_residuals = norm_residuals(obs_vis, pred_rel_vis)
            return pd.Series([rel_residuals, norm_rel_residuals])

        rel_df[residual_cols] = rel_df.apply(lambda row: calc_residuals(row), axis=1)
        rel_df.set_index(idxs, inplace=True)

        if pkl_df:
            rel_df.to_pickle(out_fn)

    return rel_df
