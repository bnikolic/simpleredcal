"""Plotting utility functions"""


import os

import numpy
from matplotlib import pyplot as plt


def plot_red_vis(cdata, redg, vis_type='amp', figsize=(13, 4)):
    """Pot visibility amplitudes or phases, grouped by redundant type

    :param cdata: Grouped visibilities with format consistent with redg
    :type cdata: ndarray
    :param redg: Grouped baselines, as returned by groupBls
    :type redg: ndarray
    :param vis_type: Plot either visibility amplitude or phase {'amp', 'phase'}
    :type vis_type: str
    """
    vis_calc = {'amp':numpy.abs, 'phase': numpy.angle}
    bl_id_seperations = numpy.unique(redg[:, 0], return_index=True)[1][1:]
    fig, ax = plt.subplots(figsize=figsize)
    ax.matshow(vis_calc[vis_type](cdata), aspect='auto')
    for bl_id_seperation in bl_id_seperations:
        plt.axvline(x=bl_id_seperation, color='white', linestyle='-.', linewidth=1)
    ax.grid(False)
    ax.set_xlabel('Baseline ID')
    ax.set_ylabel('Time Integration')
    plt.show()


def cplot(carr, figsize=(12,8), split_ax=False, save_plot=False, save_dir='plots',
          **kwargs):
    """Plot real and imaginary parts of complex array on same plot

    :param carr: Complex 1D array
    :type carr: ndarray
    :param figsize: Figure size
    :type figsize: tuple
    :param split_ax: Split real and imag components onto separate axes?
    :type split_ax: bool
    :param save_plot: Save plot?
    :type save_plot: bool
    :param save_dir: Path of directory to save plots
    :type save_dir: str
    """
    if not split_ax:
        plt.figure(figsize=figsize)
        plt.plot(carr.real, label='Real')
        plt.plot(carr.imag, label='Imag')
        for k, v in kwargs.items():
            getattr(plt, k)(v)
        plt.legend()
    else:
        fig, ax = plt.subplots(nrows=2, sharex=True, figsize=figsize)
        ax[0].plot(carr.real)
        ax[1].plot(carr.imag)
        ax[0].set_ylabel('Real')
        ax[1].set_ylabel('Imag')
        plt.xlabel('Baseline')
        if 'title' in kwargs.keys():
            fig.suptitle(kwargs['title'])
    if save_plot:
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        plt.savefig('{}/vis.png'.format(save_dir))
    plt.show()


ylab_dict = {'nit': 'number of iterations', 'fun': 'log-likelihood'}


def plot_res(df, ycol, logy=False, ylim=None, figsize=(12,8)):
    """Plot attribute of calibration results

    :param df: Results dataframe
    :type df: DataFrame
    :param ycol: Attribute to plot
    :type ycol: str
    :param logy: Bool to make y-axis scale logarithmic
    :type logy: bool
    :param figsize: Figure size of plot
    :type figsize: tuple
    """
    ax = df[ycol].plot(figsize=figsize, ylim=ylim)
    ylog = ''
    if logy:
        ax.set_yscale('log')
        ylog = 'Log '
    ax.set_ylabel((ylog+ylab_dict[ycol]).capitalize())
    plt.show()


def plot_res_grouped(df, ycol, group_by='success', logy=False, figsize=(12,8)):
    """Plot attribute of calibration results, grouped by another attribute

    :param df: Results dataframe
    :type df: DataFrame
    :param ycol: Attribute to plot
    :type ycol: str
    :param group_by: Column by which to group scatter points
    :type group_by: str
    :param logy: Bool to make y-axis scale logarithmic
    :type logy: bool
    :param figsize: Figure size of plot
    :type figsize: tuple
    """
    idx_dict = {k:i for i, k in enumerate(df.index.values)}

    x1 = [idx_dict[i] for i in df[df[group_by]][ycol].index.values]
    y1 = df[df[group_by]][ycol].values

    x2 = [idx_dict[i] for i in df[~df[group_by]][ycol].index.values]
    y2 = df[~df[group_by]][ycol].values

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(x1, y1, s=4, alpha=0.5, label=group_by)
    ax.scatter(x2, y2, s=4, color='orange', zorder=1, label='~'+group_by)
    ylog = ''
    if logy:
        ax.set_yscale('log')
        ylog = 'Log '
    ax.set_ylabel((ylog+ylab_dict[ycol]).capitalize())
    plt.legend()
    plt.show()

    if ~df['success'].any():
        pgbmax = round(numpy.max(df[ycol][~df['success']].values), 3)
    else:
        pgbmax = 'n/a - all minimizations were succesful'

    print('Max {} for minimizations with {}=False: {}\n'.format(ylab_dict[ycol], \
          group_by, pgbmax))

    print('Max {} for minimizations with {}=True: {}'.format(ylab_dict[ycol], \
          group_by, round(numpy.max(df[ycol][df['success']].values), 3)))
