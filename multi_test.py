"""Trying to run multiprocessing"""


import multiprocessing
import pymp

from red_likelihood import group_data, doRelCal
from red_utils import find_zen_file


filename = find_zen_file(2458098.43869)
pol = 'ee'
freq_channel = 300
bad_ants = [0, 2, 11, 24, 50, 53, 54, 67, 69, 98, 122, 136, 139]

hdraw, cRedG, cData = group_data(filename, pol, freq_channel, bad_ants)
cData = cData[:2, :]


def test_pymp():
    no_cores = multiprocessing.cpu_count()

    with pymp.Parallel(no_cores) as p:
        for time_int in p.xrange(cData.shape[0]):
            res = doRelCal(cRedG, cData[time_int, :], distribution='cauchy')
            print(time_int, res['fun'])


def multiprocessing_func(redg, vis_data, time_int, distribution='cauchy'):
    obsvis = vis_data[time_int, :]
    res = doRelCal(redg, obsvis, distribution=distribution)
    print('process done')
    return res            


def test_multi_process():
    processes = []
    for time_int in range(cData.shape[0]):
        p = multiprocessing.Process(target=multiprocessing_func, \
                                    args=(cRedG, cData, time_int))
        processes.append(p)

    for p in processes:
        p.start()
        p.join()

    print('Done')              


def test_multi_pool():
    pool = multiprocessing.Pool()
    for time_int in range(cData.shape[0]):
        pool.apply_async(multiprocessing_func, args=(cRedG, cData, time_int))
    pool.close()
    pool.join()


if __name__ == '__main__':
    test_multi_process()
