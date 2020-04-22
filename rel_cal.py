"""Batch relative calibration of visibilities across frequencies and time"""

import os
import pickle

import pandas as pd
import numpy

from red_likelihood import doRelCal, group_data
from red_utils import find_zen_file, get_bad_ants


def main():
    pol = 'ee'
    time_int_range = numpy.arange(2)
    freq_chan_range = numpy.arange(300, 302)

    filename = find_zen_file(2458098.43869)
    bad_ants = get_bad_ants(filename)
    hdraw, cRedG, cData = group_data(filename, pol, freq_chan_range, bad_ants)

    res_dict = {}
    with open(os.devnull, 'w'):
        for iter_dims in numpy.ndindex(cData[:, time_int_range, :].shape[:2]):
            res_rel = doRelCal(cRedG, cData[iter_dims], distribution='cauchy')
            res_dict[iter_dims] = res_rel

    df = pd.DataFrame.from_dict(res_dict, orient='index')
    df[['freq', 'time_int']] = pd.DataFrame(df.index.tolist(), index=df.index)
    df.reset_index(drop=True, inplace=True)
    df.set_index(['freq', 'time_int'], inplace=True)
    df.to_pickle('./test_res_df.pkl')
    print('Relative calibration results saved to dataframe')


if __name__ == '__main__':
    main()
