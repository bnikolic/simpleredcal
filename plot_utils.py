"""Plotting utility functions"""


import os

import matplotlib
import numpy
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib import ticker


def plot_red_vis(cdata, redg, vis_type='amp', figsize=(13, 4)):
    """Pot visibility amplitudes or phases, grouped by redundant type

    :param cdata: Grouped visibilities with format consistent with redg
    :type cdata: ndarray
    :param redg: Grouped baselines, as returned by groupBls
    :type redg: ndarray
    :param vis_type: Plot either visibility amplitude or phase {'amp', 'phase'}
    :type vis_type: str
    :param figsize: Figure size of plot
    :type figsize: tuple
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


def clip_ylimtop(df, col, clip_pctile):
    """Determine the top ylimit, with clipping applied, to better plot the
    selected column in a dataframe

    :param df: Results dataframe
    :type df: DataFrame
    :param col: Column to plot
    :type col: str
    :param clip_pctile: Percentile to clip the data
    :type clip_pctile: int, float
    """
    if col == 'nit':
        ylimtop = numpy.ceil(numpy.nanpercentile(df[col].values, clip_pctile))
    else:
        rnd_base = 10**-numpy.floor(numpy.log10(numpy.median(df[col].values)))
        ylimtop = numpy.ceil(numpy.nanpercentile(df[col].values, \
                             clip_pctile)*rnd_base)/rnd_base
    return ylimtop


def plot_res(df, col, logy=False, clip=False, clip_pctile=99, ylim=None, \
             ylabel='', title=None, figsize=(12,8)):
    """Plot attribute of calibration results

    :param df: Results dataframe
    :type df: DataFrame
    :param col: Attribute to plot
    :type col: str
    :param logy: Bool to make y-axis scale logarithmic
    :type logy: bool
    :param clip: Whether to clip the data shown in the plot, according to
    clip_pctile, to better show the data
    :type clip: bool
    :param clip_pctile: Percentile to clip the data
    :type clip_pctile: int, float
    :param ylim: Set the bottom and dtop ylimits
    :type ylim: int, float, None
    :param ylabel: ylabel of the plot
    :type ylabel: str
    :param values: Title of plot
    :type values: str, None
    :param figsize: Figure size of plot
    :type figsize: tuple
    """
    ax = df[col].plot(figsize=figsize, ylim=ylim)
    if clip:
        ax.set_ylim(bottom=0, top=clip_ylimtop(df, col, clip_pctile))
    ylog = ''
    if logy:
        ax.set_yscale('log')
        ylog = 'Log '
    if col in ylab_dict.keys():
        ylabel = ylab_dict[col]
    ax.set_ylabel((ylog+ylabel).capitalize())
    ax.set_title(title)
    plt.show()


def plot_res_grouped(df, col, group_by='success', logy=False, ylabel='', \
                     figsize=(12,8)):
    """Plot attribute of calibration results, grouped by another attribute

    :param df: Results dataframe
    :type df: DataFrame
    :param col: Attribute to plot
    :type col: str
    :param group_by: Column by which to group scatter points
    :type group_by: str
    :param logy: Bool to make y-axis scale logarithmic
    :type logy: bool
    :param ylabel: ylabel of the plot
    :type ylabel: str
    :param figsize: Figure size of plot
    :type figsize: tuple
    """
    idx_dict = {k:i for i, k in enumerate(df.index.values)}
    x1 = [idx_dict[i] for i in df[df[group_by]][col].index.values]
    x2 = [idx_dict[i] for i in df[~df[group_by]][col].index.values]
    y1 = df[df[group_by]][col].values
    y2 = df[~df[group_by]][col].values

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(x1, y1, s=4, alpha=0.5, label=group_by)
    ax.scatter(x2, y2, s=4, color='orange', zorder=1, label='~'+group_by)
    ylog = ''
    if logy:
        ax.set_yscale('log')
        ylog = 'Log '
    if col in ylab_dict.keys():
        ylabel = ylab_dict[col]
    ax.set_ylabel((ylog+ylabel).capitalize())
    plt.legend()
    plt.show()

    if ylabel == '':
        ylabel = col
    if (~df['success']).any():
        pgbmax = round(numpy.max(df[col][~df['success']].values), 3)
    else:
        pgbmax = 'n/a - all minimizations were succesful'
    print('Max {} for minimizations with {}=False: {}\n'.format(ylabel, \
          group_by, pgbmax))
    print('Max {} for minimizations with {}=True: {}'.format(ylabel, \
          group_by, round(numpy.max(df[col][df['success']].values), 3)))


def plot_res_heatmap(df, value, index='time_int', columns='freq', clip=False, \
                     clip_pctile=99, vmin=None, vmax=None, center=None, \
                     cmap=sns.cm.rocket_r, figsize=(11,7)):
    """Plot heatmap of results of redundant calibration

    :param df: Results dataframe
    :type df: DataFrame
    :param value: Values of pivoted dataframe
    :type value: str
    :param index: Index of pivoted dataframe
    :type index: str
    :param columns: Columns of pivoted dataframe
    :type columns: str
    :param clip: Whether to clip the data shown in the plot, according to
    clip_pctile, to better show the data
    :type clip: bool
    :param clip_pctile: Percentile to clip the data
    :type clip_pctile: int, float
    :param vmax: Minimum value of heatmap
    :type vmax: float
    :param vmax: Maximum value of heatmap
    :type vmax: float
    :param center: Value at which to center the colourmap
    :type center: float
    :param cmap: Colour mapping from data values to colour space
    :type cmap: str, matplotlib colormap name or object, list
    :param figsize: Figure size of plot
    :type figsize: tuple
    """
    piv = pd.pivot_table(df, values=value, index=index, columns=columns)
    fig, ax = plt.subplots(figsize=figsize)
    if clip:
        vmax=clip_ylimtop(df, value, clip_pctile)
    if (df[value].values < 0).any():
        cmap = 'bwr' # divergent colouring
        center = 0
    ax = sns.heatmap(piv, vmin=vmin, vmax=vmax, cmap=cmap, center=center)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(50))
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter(useOffset=-50))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(5))
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
    plt.tight_layout()
    plt.show()


def clipped_heatmap(arr, ylabel, xlabel='Frequency channel', clip_pctile=97, \
                    xbase=50, ybase=5, center=None, cmap=sns.cm.rocket_r, \
                    figsize=(14,7)):
    """Plots heatmap of visibility-related data, with vmax set as a percentile
    of the dataframe

    :param arr: 2D array
    :type arr: ndarray
    :param ylabel: ylabel of the plot
    :type ylabel: str
    :param xlabel: xlabel of the plot
    :type xlabel: str
    :param clip_pctile: Percentile to clip the data shown in the heatmap - used
    to set vmax
    :type clip_pctile: int, float
    :param xbase: x axis limits and tickets are multiples of this value
    :type xbase: int
    :param ybase: x axis limits and tickets are multiples of this value
    :type ybase: int
    :param center: Value at which to center the colourmap
    :type center: float
    :param cmap: Colour mapping from data values to colour space
    :type cmap: str, matplotlib colormap name or object, list
    :param figsize: Figure size of plot
    :type figsize: tuple

    :return: Tuple of Figure and Axes objects, as usually returned by
    matplotlib.pyplot.subplots
    :rtype: tuple
    """

    # clip on both the bottom and top ends of the array
    vmin = None
    if (arr < 0).any():
        clip_pctile_b = (100 - clip_pctile)/2
        clip_pctile = clip_pctile - clip_pctile_b
        vmin = numpy.floor(numpy.nanpercentile(arr, clip_pctile_b)*100)/100
        cmap = 'bwr' # divergent colouring
        center = 0

    vmax = numpy.ceil(numpy.nanpercentile(arr, clip_pctile)*100)/100

    fig, ax = plt.subplots(figsize=figsize)
    ax = sns.heatmap(arr, vmax=vmax, vmin=vmin, cmap=cmap, center=center)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(xbase))
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.yaxis.set_major_locator(ticker.MultipleLocator(ybase))
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return fig, ax


def antpos_map(values, antpos, title=None, std_rng=2, center=None, \
               cmap='bwr', figsize=(10, 8)):
    """Scatter plot of values attributed to antennas, according to their
    physical positions

    :param values: Values to determine colour of antenna scatter points
    :type values: ndarray
    :param values: Filtered (bad ants removed) antenna positions
    :type values: dict
    :param values: Title of plot
    :type values: str, None
    :param std_rng: Number of standard deviations for the values to set the vmin
    and vmax of the colourmap
    :type std_rng: int, float
    :param center: Value at which to center the colourmap
    :type center: float
    :param cmap: Colour mapping from data values to colour space
    :type cmap: str, matplotlib colormap name or object, list
    :param figsize: Figure size of plot
    :type figsize: tuple
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    vrng = numpy.ceil(numpy.std(values)*std_rng*10)/10
    if center is not None:
        vmin = center-vrng
        vmax = center+vrng
    else:
        vmin = None
        vmax = None
        center = numpy.mean(values)
    im = ax.scatter(numpy.array(list(antpos.values()))[:,0], \
                    numpy.array(list(antpos.values()))[:,1], \
                    c=values, s=800, cmap=cmap, vmin=vmin, \
                    vmax=vmax)
    # get RGBA colours of individual points
    rgba = im.to_rgba(values)
    # ITU-R 601-2 luma transform to greyscale
    c_lin = 0.299*rgba[:, 0] + 0.587*rgba[:, 1] + 0.114*rgba[:, 2]
    for i, (ant_no, pos) in enumerate(antpos.items()):
        if c_lin[i] > 0.87:
            colour='black'
        else:
            colour='white'
        ax.text(pos[0], pos[1], str(ant_no), va='center', ha='center', \
                color=colour)
    ax.set_xlabel("East-West [m]")
    ax.set_ylabel("North-South [m]")
    ax.set_title(title);
    ax.axis('equal')
    fig.colorbar(im, ax=ax)
    plt.tight_layout()
