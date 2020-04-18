"""Trying to run multiprocessing with pymp..."""


import multiprocessing
import pymp

from red_likelihood import group_data, doRelCal


mdm_fn = '/Users/matyasmolnar/Downloads/HERA_Data/robust_cal/zen.2458098.43869.HH.uvh5'
if os.path.exists(mdm_fn):
    filename = mdm_fn
else:
    filename = './zen.2458098.43869.HH.uvh5'

pol = 'ee'
freq_channel = 300
bad_ants = [0, 2, 11, 24, 50, 53, 54, 67, 69, 98, 122, 136, 139]


def main():
    no_cores = multiprocessing.cpu_count()

    hdraw, cRedG, cData = group_data(filename, pol, freq_channel, bad_ants)
    cData = cData[:4, :]

    with pymp.Parallel(no_cores) as p:
        for time_int in p.xrange(cData.shape[0]):
            res = doRelCal(cRedG, cData[time_int, :], distribution='cauchy')
            print(time_int, res['fun'])


if __name__ == '__main__':
    main()
