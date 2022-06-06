"""Plotting utility functions"""


import os

import matplotlib
import numpy
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib import ticker
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable


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
        ax.axvline(x=bl_id_seperation, color='white', linestyle='-.', linewidth=1)

    ax.grid(False)
    ax.set_xlabel('Baseline ID')
    ax.set_ylabel('Time Integration')

    fig.tight_layout()
    plt.show()


def cplot(carr, figsize=(12, 8), split_ax=False, alpha=None, savefig=None, **kwargs):
    """Plot real and imaginary parts of complex array on same plot

    :param carr: Complex 1D array
    :type carr: ndarray
    :param figsize: Figure size
    :type figsize: tuple
    :param split_ax: Split real and imag components onto separate axes?
    :type split_ax: bool
    :param alpha: alpha for lineplots
    :type alpha: float
    :param savefig: Path of directory to save plots
    :type savefig: str
    """
    if not split_ax:
        fig, ax = plt.subplots(figsize=figsize)

        ax.plot(carr.real, label=r'$\mathfrak{Re}$', alpha=alpha)
        ax.plot(carr.imag, label=r'$\mathfrak{Im}$', alpha=alpha)

        for k, v in kwargs.items():
            getattr(plt, k)(v)

        ax.legend(loc='best')
    else:
        fig, ax = plt.subplots(nrows=2, sharex=True, figsize=figsize)

        ax[0].plot(carr.real, alpha=alpha)
        ax[1].plot(carr.imag, alpha=alpha)

        ax[0].set_ylabel('Real')
        ax[1].set_ylabel('Imag')
        ax[1].set_xlabel('Baseline')

        if 'title' in kwargs.keys():
            fig.suptitle(kwargs['title'])

    fig.tight_layout()
    if savefig is not None:
        plt.savefig(savefig, bbox_inches='tight')
    plt.show()


ylab_dict = {'nit': 'number of iterations', 'fun': 'negative log-likelihood'}


def clip_ylim(df, col, clip_pctile, pos='top'):
    """Determine the top or bottom ylimit, with clipping applied, to better plot the
    selected column in a dataframe

    :param df: Results dataframe
    :type df: DataFrame
    :param col: Column to plot
    :type col: str
    :param clip_pctile: Percentile to clip the data
    :type clip_pctile: int, float
    :param pos: {"top", "bottom"} to either ceil or floor the limit
    :type pos: str
    """
    pos_dict = {'top':numpy.ceil, 'bottom':numpy.floor}
    values = df[col].values
    if col == 'nit':
        ylim = pos_dict[pos](numpy.nanpercentile(values, clip_pctile))
    else:
        if numpy.inf in values:
            values = values[values != numpy.inf]
        if -numpy.inf in values:
            values = values[values != -numpy.inf]
        if numpy.nan in values:
            values = values[values != numpy.nan]
        if (values < 0).any():
            ref = numpy.nanpercentile(values, 85)
        else:
            ref = numpy.nanmedian(values)
        rnd_base = 10**-numpy.floor(numpy.log10(ref))
        ylim = pos_dict[pos](numpy.nanpercentile(values, \
                             clip_pctile)*rnd_base)/rnd_base
    return ylim


def plot_res(df, col, logy=False, clip=False, clip_pctile=99, ylim=None, \
             ylabel='', title=None, figsize=(12, 8)):
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
    :param ylim: Set the bottom and top ylimits
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
        ax.set_ylim(bottom=0, top=clip_ylim(df, col, clip_pctile))
    ylog = ''
    if logy:
        ax.set_yscale('log')
        ylog = 'Log '
    if col in ylab_dict.keys():
        ylabel = ylab_dict[col]
    ax.set_ylabel((ylog+ylabel).capitalize())
    ax.set_title(title)
    plt.tight_layout()
    plt.show()


def plot_res_grouped(df, col, group_by='success', logy=False, ylabel='', \
                     figsize=(12, 8)):
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
    plt.tight_layout()
    plt.show()

    if ylabel == '':
        ylabel = col
    if (~df['success']).any():
        pgbmax = round(numpy.nanmax(df[col][~df['success']].values), 3)
    else:
        pgbmax = 'n/a - all minimizations were succesful'
    print('Max {} for minimizations with {}=False: {}\n'.format(ylabel, \
          group_by, pgbmax))
    print('Max {} for minimizations with {}=True: {}'.format(ylabel, \
          group_by, round(numpy.max(df[col][df['success']].values), 3)))


def plot_res_heatmap(df, col, index='time_int', columns='freq', clip=False, \
                     clip_pctile=99, vmin=None, vmax=None, center=None, logv=False, \
                     clip_bottom=False, cmap=sns.cm.rocket_r, xoffset=None, \
                     figsize=(11, 7)):
    """Plot heatmap of results of redundant calibration

    :param df: Results dataframe
    :type df: DataFrame
    :param col: Attribute to plot
    :type col: str
    :param index: Index of pivoted dataframe
    :type index: str
    :param columns: Columns of pivoted dataframe
    :type columns: str
    :param clip: Whether to clip the data shown in the plot, according to
    clip_pctile, to better show the data
    :type clip: bool
    :param clip_pctile: Percentile to clip the data
    :type clip_pctile: int, float
    :param vmin: Minimum value of heatmap
    :type vmin: float
    :param vmax: Maximum value of heatmap
    :type vmax: float
    :param center: Value at which to center the colourmap
    :type center: float
    :param logv: Logarithm scaling for heatmap values
    :type logv: bool
    :param clip_bottom: Clip the bottom values as well as the top ones
    :type clip_bottom: bool
    :param cmap: Colour mapping from data values to colour space
    :type cmap: str, matplotlib colormap name or object, list
    :param xoffset: offset used for x-axis of heatmap
    :type xoffset: int
    :param figsize: Figure size of plot
    :type figsize: tuple
    """
    piv = pd.pivot_table(df, values=col, index=index, columns=columns)
    fig, ax = plt.subplots(figsize=figsize)
    neg_values = (df[col].values < 0).any()
    if neg_values:
        cmap = 'bwr' # divergent colouring
        center = 0
    if clip:
        if neg_values or clip_bottom:
            clip_pctile_b = (100 - clip_pctile)/2
            clip_pctile = clip_pctile - clip_pctile_b
            vmin = clip_ylim(df, col, clip_pctile_b, pos='bottom')
        vmax = clip_ylim(df, col, clip_pctile, pos='top')
        # vmax = numpy.ceil(numpy.nanpercentile(df[col].values, clip_pctile)*100)/100
        # vmin = numpy.floor(numpy.nanpercentile(df[col].values, clip_pctile_b)*100)/100

    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-2, 2))

    if logv:
        norm = LogNorm(vmin=vmin, vmax=vmax)
        ax = sns.heatmap(piv, vmin=vmin, vmax=vmax, cmap=cmap, center=center, \
                         cbar_kws={'format': formatter}, norm=norm)
    else:
        ax = sns.heatmap(piv, vmin=vmin, vmax=vmax, cmap=cmap, center=center,
                         cbar_kws={'format': formatter})

    # all_x = numpy.unique(df.reset_index()[columns].values)
    # xticks = numpy.arange(numpy.min(all_x), numpy.max(all_x), 50)
    # ax.set_xticks(xticks)
    # ax.set_xticklabels(xticks)
    if xoffset is None:
        xoffset = df.index.get_level_values(level=columns).unique().values[0]
    ax.xaxis.set_major_locator(ticker.MultipleLocator(50))
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter(useOffset=-xoffset))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(5))
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
    plt.tight_layout()
    plt.show()


def clipped_heatmap(arr, ylabel, xlabel='Frequency channel', clip_pctile=97, \
                    xbase=50, ybase=5, vmin=None, center=None, cmap=sns.cm.rocket_r, \
                    sci_format=False, clip_rnd=100, retn_vlims=False, xoffset=-50,
                    cbar_lab=None, figsize=(14, 7)):
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
    :param vmin: Minimum value of heatmap
    :type vmin: float
    :param center: Value at which to center the colourmap
    :type center: float
    :param cmap: Colour mapping from data values to colour space
    :type cmap: str, matplotlib colormap name or object, list
    :param sci_format: Use scientific notation in heatmap colorbar
    :type sci_format: bool
    :param clip_rnd: Round to nearest for clipping
    :param clip_rnd: int, float
    :param retn_vlims: Return vmin and vmax too
    :type retn_vlims: bool
    :param xoffset: offset used for x-axis of heatmap
    :type xoffset: int
    :param cbar_lab: Colorbar label
    :type cbar_lab: str
    :param figsize: Figure size of plot
    :type figsize: tuple

    :return: Tuple of Figure and Axes objects, as usually returned by
    matplotlib.pyplot.subplots
    :rtype: tuple
    """

    # clip on both the bottom and top ends of the array
    if vmin is None and (arr < 0).any():
        clip_pctile_b = (100 - clip_pctile)/2
        clip_pctile = clip_pctile - clip_pctile_b
        vmin = numpy.floor(numpy.nanpercentile(arr, clip_pctile_b)*clip_rnd)/clip_rnd
        cmap = 'bwr' # divergent colouring
        center = 0

    vmax = numpy.ceil(numpy.nanpercentile(arr, clip_pctile)*clip_rnd)/clip_rnd

    if sci_format:
        formatter = ticker.ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((-2 ,2))
    else:
        formatter = None

    fig, ax = plt.subplots(figsize=figsize)
    ax = sns.heatmap(arr, vmax=vmax, vmin=vmin, cmap=cmap, center=center, \
                     cbar_kws={'format': formatter, 'label': cbar_lab})
    ax.xaxis.set_major_locator(ticker.MultipleLocator(xbase))
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter(useOffset=xoffset))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(ybase))
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    retn = (fig, ax)
    if retn_vlims:
        retn += (vmin, vmax)
    return retn


def df_heatmap(df, xlabel=None, ylabel=None, title=None, xbase=None, ybase=None, \
               center=None, vmin=None, vmax=None, sci_format=True, cmap=sns.cm.rocket_r, \
               figsize=(11, 7)):
    """Plots heatmap of visibility-related data, with vmax set as a percentile
    of the dataframe

    :param df: Dataframe to be plotted
    :type df: DataFrame
    :param xlabel: xlabel of the plot
    :type xlabel: str
    :param ylabel: ylabel of the plot
    :type ylabel: str
    :param title: title of the plot
    :type ylabel: str
    :param xbase: x axis limits and tickets are multiples of this value
    :type xbase: int
    :param ybase: x axis limits and tickets are multiples of this value
    :type ybase: int
    :param center: Value at which to center the colourmap
    :type center: float
    :param vmin: Minimum value of heatmap
    :type vmin: float
    :param vmax: Maximum value of heatmap
    :type vmax: float
    :param sci_format: Use scientific notation in heatmap colorbar
    :type sci_format: bool
    :param cmap: Colour mapping from data values to colour space
    :type cmap: str, matplotlib colormap name or object, list
    :param figsize: Figure size of plot
    :type figsize: tuple
    """

    fig, ax = plt.subplots(figsize=figsize)

    if sci_format:
        formatter = ticker.ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((-2 ,2))
    else:
        formatter = None

    ax = sns.heatmap(df, cmap=cmap, center=center, vmin=vmin, vmax=vmax, \
                     cbar_kws={'format': formatter})
    if xbase is not None:
        ax.xaxis.set_major_locator(ticker.MultipleLocator(xbase))
    xoffset = df.columns.values[0]
    if xoffset != 1:
        ax.xaxis.set_major_formatter(ticker.ScalarFormatter(useOffset=-df.columns.values[0]))
    if ybase is not None:
        ax.yaxis.set_major_locator(ticker.MultipleLocator(ybase))
        ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    plt.tight_layout()
    plt.show()


def antpos_map(values, flt_ant_pos, title=None, std_rng=2, center=None, \
               cmap='bwr', figsize=(10, 8)):
    """Scatter plot of values attributed to antennas, according to their
    physical positions

    :param values: Values to determine colour of antenna scatter points
    :type values: ndarray
    :param flt_ant_pos: Filtered dict of antenna positions. See flt_ant_coords
    :type flt_ant_pos: dict
    :param title: Title of plot
    :type title: str, None
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
    im = ax.scatter(numpy.array(list(flt_ant_pos.values()))[:,0], \
                    numpy.array(list(flt_ant_pos.values()))[:,1], \
                    c=values, s=800, cmap=cmap, vmin=vmin, \
                    vmax=vmax)
    # get RGBA colours of individual points
    rgba = im.to_rgba(values)
    # ITU-R 601-2 luma transform to greyscale
    c_lin = 0.299*rgba[:, 0] + 0.587*rgba[:, 1] + 0.114*rgba[:, 2]
    for i, (ant_no, pos) in enumerate(flt_ant_pos.items()):
        if c_lin[i] > 0.87:
            colour='black'
        else:
            colour='white'
        ax.text(pos[0], pos[1], str(ant_no), va='center', ha='center', \
                color=colour)
    ax.set_xlabel('East-West [m]')
    ax.set_ylabel('North-South [m]')
    ax.set_title(title)
    ax.axis('equal')
    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.show()


def flagged_hist(values, flags, xlabel=None, lower_cut=None, upper_cut=None, \
                 bin_width=None, hist_start=0, ylim=None, logy=False, savefig=None, \
                 figsize=(8, 5)):
    """Histogram for flagged and unflagged data on the same plot, with option
    for first and last bins to include outliers

    :param values: Values to plot
    :type values: ndarray
    :param values: Flags to apply to values
    :type values: ndarray
    :param xlabel: xlabel of the plot
    :type xlabel: str
    :param lower_cut: lower cut-off to combine outliers into a single column
    :type lower_cut: int, float
    :param upper_cut: upper cut-off to combine outliers into a single column
    :type upper_cut: int, float
    :param bin_width: histogram bin width
    :type bin_width: int, float
    :param hist_start: x-start of histogram
    :type hist_start: int, float
    :param ylim: Set the bottom and top ylimits
    :type ylim: int, float, None
    :param logy: Bool to make y-axis scale logarithmic
    :type logy: bool
    :param savefig: Path to save figure
    :type savefig: str
    :param figsize: Figure size of plot
    :type figsize: tuple
    """
    fig, ax = plt.subplots(figsize=figsize)

    if upper_cut is not None:
        hist_end = upper_cut
    else:
        hist_end = numpy.nanpercentile(values, 95)
        rnd_base = 10**-numpy.floor(numpy.log10(hist_end))
        hist_end = numpy.ceil(hist_end*rnd_base)/rnd_base

    if bin_width is None:
        bin_width = (hist_end - hist_start) / 50

    if lower_cut is not None:
        hist_start = lower_cut

    bin_range = numpy.arange(hist_start, hist_end+2*bin_width, bin_width)
    # precision errors adding an extra bin if bin_width is small
    if int((hist_end - hist_start)/bin_width) + 2 < bin_range.size:
        bin_range = bin_range[:-1]

    _, _, patchesu = plt.hist(values[~flags], range=(hist_start, hist_end+bin_width), \
                              bins=bin_range, density=False, alpha=0.65, label='Unflagged')
    _, _, patchesf = plt.hist(values[flags], range=(hist_start, hist_end+bin_width), \
                              bins=bin_range, density=False, alpha=0.65, label='Flagged')

    if lower_cut is not None:
        n_lower_outliersu = (values[~flags] < lower_cut).sum()
        patchesu[0].set_height(patchesu[0].get_height() + n_lower_outliersu)
        patchesu[0].set_facecolor('m')
        patchesu[0].set_label('Unflagged lower outliers')

        n_lower_outliersf = (values[flags] < lower_cut).sum()
        patchesf[0].set_height(patchesf[0].get_height() + n_lower_outliersf)
        patchesf[0].set_facecolor('g')
        patchesf[0].set_label('Flagged lower outliers')

    if upper_cut is not None:
        n_upper_outliersu = (values[~flags] > upper_cut).sum()
        patchesu[-1].set_height(patchesu[-1].get_height() + n_upper_outliersu)
        patchesu[-1].set_facecolor('m')
        patchesu[-1].set_label('Unflagged upper outliers')

        n_upper_outliersf = (values[flags] > upper_cut).sum()
        patchesf[-1].set_height(patchesf[-1].get_height() + n_upper_outliersf)
        patchesf[-1].set_facecolor('g')
        patchesf[-1].set_label('Flagged upper outliers')

    plt.xlabel(xlabel)
    plt.ylim(ylim)
    if logy:
        ax.set_yscale('log')
    plt.legend(loc='best', framealpha=0.5)
    plt.ticklabel_format(axis='x', style='sci', scilimits=(-1, 1))
    plt.tight_layout()
    if savefig is not None:
        plt.savefig(savefig, bbox_inches='tight')
    plt.show()


def arr_pcmesh(*args, xlabel=None, ylabel=None, clabel=None, title=None, vmin=None, vmax=None, \
               xlim=None, ylim=None, extend=None, cmap=sns.cm.rocket_r, sci_fmt=True, savefig=None, \
               rtn_fig_ax=False, figsize=(7, 5), dpi=125):
    """Plot pcolormesh of array

    :param args: [X, Y], C: X, Y are the coordinates of the corners and C is the
    scalar 2D array whose values are colour-mapped
    :type args: ndarray
    :param xlabel, ylabel, clabel: labels of plot and colourbar
    :type xlabel, ylabel, clabel: str
    :param title: title of the plot
    :type title: str
    :param vmin, vmax: colourbar range
    :type vmin, vmax: float
    :param xlim, ylim: plot limits
    :type xlim, ylim: tuple
    :param extend: how to extend colourbar
    :type extend: str
    :param cmap: colour mapping from data values to colour space
    :type cmap: str, matplotlib colormap name or object, list
    :param sci_fmt: scientific notation for colourbar ticks
    :type sci_fmt: bool
    :param savefig: path to save figure
    :type savefig: str
    :param rtn_fig_ax: return figure and ax objects
    :type rtn_fig_ax: bool
    :param figsize: figure size of plot
    :type figsize: tuple
    :param dpi: dpi
    :type dpi: int
    """
    if savefig is not None:
        # ensure good resolution for rasterized saved figure
        dpi = 600

    fig, ax = plt.subplots(figsize=(7, 5), dpi=dpi)

    im = ax.pcolormesh(*args, cmap=cmap, vmin=vmin, vmax=vmax, \
                       rasterized=savefig is not None)

    ax.invert_yaxis()

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='4%', pad=0.25)
    cbar = fig.colorbar(im, cax=cax, orientation='vertical', extend=extend, label=clabel)

    if sci_fmt:
        cbar.formatter.set_scientific(True)
        cbar.formatter.set_powerlimits((0, 0))
        cbar.ax.yaxis.set_offset_position('left')
        # adjust position of scientific notation as max extend hides annotation
        if extend not in (None, 'min'):
            cbar.ax.yaxis.get_offset_text().set_visible(False)  # turn off as extend='max' hides the offset
            if vmax is None:
                vmax = args[0].max()
            cbar.ax.text(0.7, 1.06, rf'$\times 10^{{{int(numpy.floor(numpy.log10(vmax)))}}}$', \
                         va='bottom', ha='center', transform=cax.transAxes)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if not rtn_fig_ax:
        fig.tight_layout()
        if savefig is not None:
            plt.savefig(savefig, bbox_inches='tight')
        plt.show()
    else:
        return fig, ax
