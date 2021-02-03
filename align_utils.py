"""Utility functions for alignment of dataframes"""


import pandas as pd
import numpy

from red_utils import find_deg_df, find_nearest, find_rel_df, match_lst


# Note that 2458109 has 1 fewer antennas, as antenna 14 is flagged
idr2_jds = [2458098, 2458099, 2458101, 2458102, 2458103, 2458104, 2458105, \
            2458106, 2458107, 2458108, 2458109, 2458110, 2458111, 2458112, \
            2458113, 2458114, 2458115, 2458116, 2458140]


def align_df(df_type, JD_time, JD_comp, dir_path, ndist, pol, JD_anchor=2458099):
    """Build the dataframe on a separate JD that is aligned in LAST with that
    on jd_time (e.g. rel_df or deg_df) - due to offset in LAST, this requires
    the concatenation of two separate dataframes.

    :param df_type: What dataframes are being joined {"rel", "deg"}
    :type df_type: str
    :param JD_time: Fractional Julian date of dataframe we wish to be aligned to
    :type JD_time: float, str
    :param JD_comp: JD day to align
    :type JD_comp: str
    :param dir_path: Directory in which dataframes are located
    :type dir_path: str
    :param ndist: Noise distribution for calibration {"cauchy", "gaussian"}
    :type ndist: str
    :param pol: Polarization of data
    :type pol: str
    :param JD_anchor: JD of anchor day used in degenerate comparison; only if
    df_type = "deg"
    :type JD_anchor: int

    :return: Bad antennas
    :rtype: ndarray
    """
    # find dataset from specified JD that contains visibilities at the same LAST
    JD_timea = match_lst(JD_time, JD_comp)

    # aligning datasets in LAST
    last_df = pd.read_pickle('jd_lst_map_idr2.pkl')
    last1 = last_df[last_df['JD_time'] == float(JD_time)]['LASTs'].values[0]
    last2 = last_df[last_df['JD_time'] == float(JD_timea)]['LASTs'].values[0]
    _, offset = find_nearest(last2, last1[0])
    next_row = numpy.where(last_df['JD_time'] == JD_timea)[0][0] + 1
    JD_timeb = last_df.iloc[next_row]['JD_time']

    if df_type == 'rel':
        tidx = 'time_int'
        indices = ['freq', tidx]
        df_patha = find_rel_df(JD_timea, pol, ndist, dir_path)
        df_pathb = find_rel_df(JD_timeb, pol, ndist, dir_path)
    if df_type == 'deg':
        tidx = 'time_int1'
        indices = ['freq', tidx]
        df_patha = find_deg_df(JD_timea, pol, 'jd.{}'.format(JD_anchor), ndist, \
                               dir_path)
        df_pathb = find_deg_df(JD_timeb, pol, 'jd.{}'.format(JD_anchor), ndist, \
                               dir_path)

    dfb = pd.read_pickle(df_pathb)
    dfa = pd.read_pickle(df_patha)

    Nfreqs = dfa.index.get_level_values('freq').unique().size
    Ntints = dfa.index.get_level_values(tidx).unique().size

    dfa = dfa[dfa.index.get_level_values(tidx) >= offset]
    dfa.sort_index(level=indices, inplace=True)
    # shifting tints to align with those from JD_time
    dfa.reset_index(inplace=True)
    dfa[tidx] = numpy.tile(numpy.arange(Ntints - offset), Nfreqs)
    dfa.set_index(indices, inplace=True)

    dfb = dfb[dfb.index.get_level_values(tidx) < offset]
    dfb.sort_index(level=indices, inplace=True)
    # shifting tints to align with those from JD_time
    dfb.reset_index(inplace=True)
    dfb[tidx] = numpy.tile(numpy.arange(Ntints - offset, Ntints), Nfreqs)
    dfb.set_index(indices, inplace=True)

    # combined results dataframes that is now alinged in LAST by row number
    # with the dataframe labelled by JD_time
    df_c = pd.concat([dfa, dfb])
    df_c.sort_index(inplace=True)
    return df_c
