#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# @Author: oesteban
# @Date:   2016-01-05 11:32:01
# @Email:  code@oscaresteban.es
# @Last modified by:   oesteban
""" Helper functions for the figures in the paper """
from __future__ import print_function, division, absolute_import, unicode_literals
import os.path as op
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mriqc.classifier.data import read_dataset, zscore_dataset

def fill_matrix(matrix, width, value='n/a'):
    if matrix.shape[0] < width:
        nas = np.chararray((1, 3), itemsize=len(value))
        nas[:] = value
        matrix = np.vstack(tuple([matrix] + [nas] * (width - matrix.shape[0])))
    return matrix

def plot_raters(dataframe, site=None, ax=None, width=101,
                raters=None):
    if raters is None:
        raters = ['Marie', 'PCP_rater_2', 'PCP_rater_3']

    if site is not None:
        dataframe = dataframe.loc[dataframe.site == site]

    dataframe = dataframe[raters]
    matrix = dataframe.as_matrix()

    if matrix.shape[0] < width:
        matrix = fill_matrix(matrix, width)

    nblocks = 1
    if matrix.shape[0] > width:
        matrices = []
        nblocks = (matrix.shape[0] // width) + 1

        nas = np.chararray((width, 1), itemsize=3)
        nas[:] = 'n/a'
        for i in range(nblocks):
            if i > 0:
                matrices.append(nas)
            matrices.append(matrix[i * width:(i + 1) * width, ...])

        matrices[-1] = fill_matrix(matrices[-1], width)
        matrix = np.hstack(tuple(matrices))

    palette = {'OK': 'limegreen', 'maybe': 'gold', 'fail': 'tomato', 'n/a': 'w'}

    ax = ax if ax is not None else plt.gca()

    # ax.patch.set_facecolor('gray')
    ax.set_aspect('equal', 'box')
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())

    size = 0.75
    for (x, y), w in np.ndenumerate(matrix):
        if w == '':
            w = 'n/a'
        color = palette[w.strip()]
        rect = plt.Circle([x - size / 2, y - size / 2], size * 0.5,
                             facecolor=color, edgecolor=color)
        ax.add_patch(rect)

    ax.autoscale_view()
    ax.invert_yaxis()
    ax.set_yticklabels([])
    ax.set_yticks([])

    # Remove and redefine spines
    for side in ["top", "right", "bottom"]:
        # Toggle the spine objects
        ax.spines[side].set_color('none')
        ax.spines[side].set_visible(False)

    ax.spines["left"].set_linewidth(3)
    ax.spines["left"].set_color('dimgray')
    ax.spines["left"].set_position(('data', -1.5))
    plt.grid(b=False, which='major', linewidth=0)
    # ax.yaxis.set_label_position("right")

    ax.set_ylabel(site, fontsize=18, rotation=0)
    ylabel_y = 0.20

    if nblocks > 1:
        ylabel_y = 0.45

    ax.yaxis.set_label_coords(-0.01, ylabel_y)

    return ax

def raters_variability_plot(mdata, figsize=(22, 22),
                            width=101, out_file=None):
    sites_list = sorted(set(mdata.site.values.ravel().tolist()))
    sites_len = []
    for site in sites_list:
        sites_len.append(len(mdata.loc[mdata.site == site]))


    blocks = [(slen - 1) // width + 1 for slen in sites_len]
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(len(sites_list), 1, width_ratios=[1], height_ratios=blocks, hspace=0.05)

    for s, gsel in zip(sites_list, gs):
        ax = plt.subplot(gsel)
        plot_raters(mdata, site=s, ax=ax, width=width)

    if out_file is None:
        out_file = 'raters.svg'

    fname, ext = op.splitext(out_file)
    if ext[1:] not in ['pdf', 'svg', 'png']:
        ext = '.svg'
        out_file = fname + '.svg'

    fig.savefig(op.abspath(out_file), format=ext[1:],
                bbox_inches='tight', pad_inches=0, dpi=300)
    return fig

def plot_abide_stripplots(X, Y, figsize=(15, 80), out_file=None,
                          rating_label='rate'):
    import seaborn as sn
    sn.set(style="whitegrid")

    mdata, pp_cols = read_dataset(X, Y, rate_label=rating_label)

    mdata['database'] = ['ABIDE'] * len(mdata['site'].values.ravel())
    zscored = zscore_dataset(
            mdata, excl_columns=[rating_label, 'size_x', 'size_y', 'size_z',
                                 'spacing_x', 'spacing_y', 'spacing_z'])
    sites = list(set(mdata[rating_label].values.ravel()))

    palette = ['dodgerblue', 'darkorange']
    nrows = len(pp_cols)
    nsites = len(sites)

    fig = plt.figure(figsize=figsize)
    # ncols = 2 * (nsites - 1) + 2
    gs = GridSpec(nrows, 4, wspace=0.02)
    gs.set_width_ratios([6 * nsites, 1, 1, 6 * nsites])

    for i, colname in enumerate(pp_cols):
        ax_nzs = plt.subplot(gs[i, 0])
        axg_nzs = plt.subplot(gs[i, 1])
        axg_zsc = plt.subplot(gs[i, 2])
        ax_zsc = plt.subplot(gs[i, 3])

        # plots
        sn.stripplot(x='site', y=colname, data=mdata, hue=rating_label, jitter=0.18, alpha=.4,
                     split=True, palette=palette, ax=ax_nzs)
        sn.stripplot(x='site', y=colname, data=zscored, hue=rating_label, jitter=0.18, alpha=.4,
                     split=True, palette=palette, ax=ax_zsc)

        sn.stripplot(x='database', y=colname, data=mdata, hue=rating_label, jitter=0.18, alpha=.4,
                     split=True, palette=palette, ax=axg_nzs)
        sn.stripplot(x='database', y=colname, data=zscored, hue=rating_label, jitter=0.18,
                     alpha=.4, split=True, palette=palette, ax=axg_zsc)

        ax_nzs.legend_.remove()
        ax_zsc.legend_.remove()
        axg_nzs.legend_.remove()
        axg_zsc.legend_.remove()

        if i == nrows - 1:
            ax_nzs.set_xticklabels(ax_nzs.xaxis.get_majorticklabels(), rotation=80)
            ax_zsc.set_xticklabels(ax_zsc.xaxis.get_majorticklabels(), rotation=80)
            axg_nzs.set_xticklabels(axg_nzs.xaxis.get_majorticklabels(), rotation=80)
            axg_zsc.set_xticklabels(axg_zsc.xaxis.get_majorticklabels(), rotation=80)
        else:
            ax_nzs.set_xticklabels([])
            ax_zsc.set_xticklabels([])
            axg_nzs.set_xticklabels([])
            axg_zsc.set_xticklabels([])

        ax_nzs.set_xlabel('', visible=False)
        ax_zsc.set_xlabel('', visible=False)
        ax_zsc.set_ylabel('', visible=False)
        ax_zsc.yaxis.tick_right()

        axg_nzs.set_yticklabels([])
        axg_nzs.set_xlabel('', visible=False)
        axg_nzs.set_ylabel('', visible=False)
        axg_zsc.set_yticklabels([])
        axg_zsc.set_xlabel('', visible=False)
        axg_zsc.set_ylabel('', visible=False)


        for yt in ax_nzs.yaxis.get_major_ticks()[1:-1]:
            yt.label1.set_visible(False)

        for yt in axg_nzs.yaxis.get_major_ticks()[1:-1]:
            yt.label1.set_visible(False)

        for yt in zip(ax_zsc.yaxis.get_majorticklabels(), axg_zsc.yaxis.get_majorticklabels()):
            yt[0].set_visible(False)
            yt[1].set_visible(False)

    if out_file is None:
        out_file = 'stripplot.svg'

    fname, ext = op.splitext(out_file)
    if ext[1:] not in ['pdf', 'svg', 'png']:
        ext = '.svg'
        out_file = fname + '.svg'

    fig.savefig(op.abspath(out_file), format=ext[1:],
                bbox_inches='tight', pad_inches=0, dpi=300)
    return fig

def plot_corrmat(in_csv, out_file=None):
    import seaborn as sn
    sn.set(style="whitegrid")

    dataframe = pd.read_csv(in_csv, index_col=False, na_values='n/a', na_filter=False)
    colnames = dataframe.columns.ravel().tolist()

    for col in ['subject_id', 'site', 'qc_type']:
        try:
            colnames.remove(col)
        except ValueError:
            pass

    # Correlation matrix
    corr = dataframe[colnames].corr()
    corr = corr.dropna((0,1), 'all')

    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    # Generate a custom diverging colormap
    cmap = sn.diverging_palette(220, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    corrplot = sn.clustermap(corr, cmap=cmap, center=0., method='average', square=True, linewidths=.5)
    plt.setp(corrplot.ax_heatmap.yaxis.get_ticklabels(), rotation='horizontal')
    # , mask=mask, square=True, linewidths=.5, cbar_kws={"shrink": .5})

    if out_file is None:
        out_file = 'corr_matrix.svg'

    fname, ext = op.splitext(out_file)
    if ext[1:] not in ['pdf', 'svg', 'png']:
        ext = '.svg'
        out_file = fname + '.svg'

    corrplot.savefig(out_file, format=ext[1:], bbox_inches='tight', pad_inches=0, dpi=100)
    return corrplot


def plot_histograms(X, Y, rating_label='rate', out_file=None):
    import seaborn as sn
    sn.set(style="whitegrid")

    dataframe, pp_cols = read_dataset(X, Y, rate_label=rating_label)

    zscored = zscore_dataset(
        dataframe, excl_columns=[rating_label, 'size_x', 'size_y', 'size_z',
                                 'spacing_x', 'spacing_y', 'spacing_z'])

    colnames = [col for col in sorted(pp_cols)
                if not (col.startswith('spacing') or col.startswith('summary') or col.startswith('size'))]

    nrows = len(colnames)
    # palette = ['dodgerblue', 'darkorange']

    fig = plt.figure(figsize=(18, 2 * nrows))
    gs = GridSpec(nrows, 2, hspace=0.2)

    for i, col in enumerate(sorted(colnames)):
        ax_nzs = plt.subplot(gs[i, 0])
        ax_zsd = plt.subplot(gs[i, 1])

        sn.distplot(dataframe.loc[dataframe.rate == 0, col], norm_hist=False,
                    label='Accept', ax=ax_nzs)
        sn.distplot(dataframe.loc[dataframe.rate == 1, col], norm_hist=False,
                    label='Reject', ax=ax_nzs)
        ax_nzs.legend()

        sn.distplot(zscored.loc[zscored.rate == 0, col], norm_hist=False,
                    label='Accept', ax=ax_zsd)
        sn.distplot(zscored.loc[zscored.rate == 1, col], norm_hist=False,
                    label='Reject', ax=ax_zsd)

        alldata = dataframe[[col]].values.ravel().tolist()
        minv = np.percentile(alldata, 0.2)
        maxv = np.percentile(alldata, 99.8)
        ax_nzs.set_xlim([minv, maxv])

        alldata = zscored[[col]].values.ravel().tolist()
        minv = np.percentile(alldata, 0.2)
        maxv = np.percentile(alldata, 99.8)
        ax_zsd.set_xlim([minv, maxv])

    if out_file is None:
        out_file = 'histograms.svg'

    fname, ext = op.splitext(out_file)
    if ext[1:] not in ['pdf', 'svg', 'png']:
        ext = '.svg'
        out_file = fname + '.svg'

    fig.savefig(out_file, format=ext[1:], bbox_inches='tight', pad_inches=0, dpi=100)
    return fig
