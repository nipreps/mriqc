#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
from __future__ import print_function, division, absolute_import, unicode_literals

import nibabel as nb
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib import gridspec as mgs
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.colorbar import ColorbarBase
import seaborn as sns
from seaborn import color_palette

from .utils import DINA4_LANDSCAPE
sns.set_style("whitegrid")

class fMRIPlot(object):

    def __init__(self, func, mask, seg=None, tr=None,
                 title=None, figsize=DINA4_LANDSCAPE):
        func_nii = nb.load(func)
        self.func_data = func_nii.get_data()
        self.mask_data = nb.load(mask).get_data()

        self.ntsteps = self.func_data.shape[-1]
        self.tr = tr
        if tr is None:
            self.tr = func_nii.get_header().get_zooms()[-1]

        if seg is None:
            self.seg_data = 2 * self.mask_data
        else:
            self.seg_data = nb.load(seg).get_data()

        self.fig = plt.figure(figsize=figsize)
        if title is not None:
            self.fig.suptitle(title, fontsize=20)

        self.confounds = []
        self.spikes = []

    def add_confounds(self, data, kwargs):
        self.confounds.append((data, kwargs))

    def add_spikes(self, tsz, title=None, zscored=True):
        self.spikes.append((tsz, title, zscored))

    def plot(self):
        nconfounds = len(self.confounds)
        nspikes = len(self.spikes)
        nrows = 1 + nconfounds + nspikes

        # Create grid
        grid = mgs.GridSpec(nrows, 1, wspace=0.0, hspace=0.2,
                            height_ratios=[1] * (nrows - 1) + [3.5])

        grid_id = 0
        for tsz, name, iszs in self.spikes:
            spikesplot(tsz, title=name, outer_gs=grid[grid_id], tr=self.tr,
                       zscored=iszs)
            grid_id += 1

        if self.confounds:
            palette = color_palette("husl", nconfounds)

        for i, (tseries, kwargs) in enumerate(self.confounds):
            confoundplot(
                tseries, grid[grid_id], tr=self.tr, color=palette[i],
                **kwargs)
            grid_id += 1

        fmricarpetplot(self.func_data, self.seg_data,
                       grid[-1], tr=self.tr)

        setattr(self, 'grid', grid)
        # spikesplot_cb([0.7, 0.78, 0.2, 0.008])


def fmricarpetplot(func_data, segmentation, outer_gs, tr=None, nskip=0):
    """
    Plot "the plot"
    """
    from nilearn.signal import clean

    # Define TR and number of frames
    notr = False
    if tr is None:
        notr = True
        tr = 1.
    ntsteps = func_data.shape[-1]

    data = func_data[segmentation > 0].reshape(-1, ntsteps)
    # Detrend data
    detrended = clean(data.T, t_r=tr).T

    # Order following segmentation labels
    seg = segmentation[segmentation > 0].reshape(-1)
    seg_labels = np.unique(seg)

    # Labels meaning
    cort_gm = seg_labels[(seg_labels > 100) & (seg_labels < 200)].tolist()
    deep_gm = seg_labels[(seg_labels > 30) & (seg_labels < 100)].tolist()
    cerebellum = [255]
    wm_csf = seg_labels[seg_labels < 10].tolist()
    seg_labels = cort_gm + deep_gm + cerebellum + wm_csf

    label_id = 0
    newsegm = np.zeros_like(seg)
    for _lab in seg_labels:
        newsegm[seg == _lab] = label_id
        label_id += 1
    order = np.argsort(newsegm)

    # Define nested GridSpec
    gs = mgs.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer_gs,
                                     width_ratios=[1, 100], wspace=0.0)

    # Segmentation colorbar
    ax0 = plt.subplot(gs[0])
    ax0.set_yticks([])
    ax0.set_xticks([])

    colors1 = plt.cm.summer(np.linspace(0., 1., len(cort_gm)))
    colors2 = plt.cm.autumn(np.linspace(0., 1., len(deep_gm) + 1))[::-1,...]
    colors3 = plt.cm.winter(np.linspace(0., .5, len(wm_csf)))[::-1,...]
    cmap = LinearSegmentedColormap.from_list('my_colormap', np.vstack((colors1, colors2, colors3)))

    ax0.imshow(newsegm[order, np.newaxis], interpolation='nearest', aspect='auto',
               cmap=cmap, vmax=len(seg_labels) - 1, vmin=0)
    ax0.grid(False)
    ax0.set_ylabel('voxels')

    # Carpet plot
    ax1 = plt.subplot(gs[1])

    # Avoid segmentation faults for long acquisitions by decimating the input data
    long_cutoff = 800
    if detrended.shape[1] > long_cutoff:
        data = detrended[order, ::2]
    else:
        data = detrended[order, :]

    ax1.imshow(data, interpolation='nearest',
               aspect='auto', cmap='gray', vmin=-2, vmax=2)

    ax1.grid(False)
    ax1.set_yticks([])
    ax1.set_yticklabels([])

    # Set 10 frame markers in X axis
    interval = int(data.shape[-1] + 1) // 10
    xticks = list(
        range(0, data.shape[-1])[::interval]) + [data.shape[-1]-1]
    ax1.set_xticks(xticks)

    if notr:
        ax1.set_xlabel('time (frame #)')
    else:
        ax1.set_xlabel('time (s)')
        labels = tr * (np.array(xticks))
        if detrended.shape[1] > long_cutoff:
            labels *= 2
        ax1.set_xticklabels(['%.02f' % t for t in labels.tolist()])

    # Remove and redefine spines
    for side in ["top", "right"]:
        # Toggle the spine objects
        ax0.spines[side].set_color('none')
        ax0.spines[side].set_visible(False)
        ax1.spines[side].set_color('none')
        ax1.spines[side].set_visible(False)

    ax1.yaxis.set_ticks_position('left')
    ax1.xaxis.set_ticks_position('bottom')
    ax1.spines["bottom"].set_position(('outward', 20))
    ax1.spines["left"].set_color('none')
    ax1.spines["left"].set_visible(False)

    ax0.spines["left"].set_position(('outward', 20))
    ax0.spines["bottom"].set_color('none')
    ax0.spines["bottom"].set_visible(False)

    return [ax0, ax1], gs


def spikesplot(ts_z, outer_gs=None, tr=None, zscored=True, spike_thresh=6., title='Spike plot',
               ax=None, cmap='viridis', hide_x=True, nskip=0):
    """
    A spikes plot. Thanks to Bob Dogherty (this docstring needs be improved with proper ack)
    """

    if ax is None:
        ax = plt.gca()

    if not outer_gs is None:
        gs = mgs.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer_gs,
                                         width_ratios=[1, 100], wspace=0.0)
        ax = plt.subplot(gs[1])

    # Define TR and number of frames
    if tr is None:
        tr = 1.

    # Load timeseries, zscored slice-wise
    nslices = ts_z.shape[0]
    ntsteps = ts_z.shape[1]

    # Load a colormap
    my_cmap = get_cmap(cmap)
    norm = Normalize(vmin=0, vmax=float(nslices - 1))
    colors = [my_cmap(norm(sl)) for sl in range(nslices)]

    stem = len(np.unique(ts_z).tolist()) == 2
    # Plot one line per axial slice timeseries
    for sl in range(nslices):
        if not stem:
            ax.plot(ts_z[sl, :], color=colors[sl], lw=0.5)
        else:
            markerline, stemlines, baseline = ax.stem(ts_z[sl, :])
            plt.setp(markerline, 'markerfacecolor', colors[sl])
            plt.setp(baseline, 'color', colors[sl], 'linewidth', 1)
            plt.setp(stemlines, 'color', colors[sl], 'linewidth', 1)

    # Handle X, Y axes
    ax.grid(False)

    # Handle X axis
    last = ntsteps - 1
    ax.set_xlim(0, last)
    xticks = list(range(0, last)[::20]) + [last] if not hide_x else []
    ax.set_xticks(xticks)

    if not hide_x:
        if tr is None:
            ax.set_xlabel('time (frame #)')
        else:
            ax.set_xlabel('time (s)')
            ax.set_xticklabels(
                ['%.02f' % t for t in (tr * np.array(xticks)).tolist()])

    # Handle Y axis
    if zscored:
        ax.set_ylabel('z-score')
        zs_max = np.abs(ts_z).max()
        ax.set_ylim((-(np.abs(ts_z[:, nskip:]).max()) * 1.05,
                     (np.abs(ts_z[:, nskip:]).max()) * 1.05))

        ytick_vals = np.arange(0.0, zs_max, float(np.floor(zs_max / 2.)))
        yticks = list(
            reversed((-1.0 * ytick_vals[ytick_vals > 0]).tolist())) + ytick_vals.tolist()

        # TODO plot min/max or mark spikes
        # yticks.insert(0, ts_z.min())
        # yticks += [ts_z.max()]
        for val in ytick_vals:
            ax.plot((0, ntsteps - 1), (-val, -val), 'k:', alpha=.2)
            ax.plot((0, ntsteps - 1), (val, val), 'k:', alpha=.2)

        # Plot spike threshold
        if zs_max < spike_thresh:
            ax.plot((0, ntsteps - 1), (-spike_thresh, -spike_thresh), 'k:')
            ax.plot((0, ntsteps - 1), (spike_thresh, spike_thresh), 'k:')
    else:
        ax.set_ylabel('air sgn. intensity')
        yticks = [ts_z[:, nskip:].min(),
                  np.median(ts_z[:, nskip:]),
                  ts_z[:, nskip:].max()]

        ax.set_ylim(ts_z[:, nskip:].min() * 0.95,
                    ts_z[:, nskip:].max() * 1.05)

    if yticks:
        ax.set_yticks(yticks)
        ax.set_yticklabels(['%.02f' % y for y in yticks])
        # Plot maximum and minimum horizontal lines
        ax.plot((0, ntsteps - 1), (yticks[0], yticks[0]), 'k:')
        ax.plot((0, ntsteps - 1), (yticks[-1], yticks[-1]), 'k:')


    for side in ["top", "right"]:
        ax.spines[side].set_color('none')
        ax.spines[side].set_visible(False)

    if not hide_x:
        ax.spines["bottom"].set_position(('outward', 20))
        ax.xaxis.set_ticks_position('bottom')
    else:
        ax.spines["bottom"].set_color('none')
        ax.spines["bottom"].set_visible(False)

    ax.spines["left"].set_position(('outward', 30))
    ax.yaxis.set_ticks_position('left')

    # labels = [label for label in ax.yaxis.get_ticklabels()]
    # labels[0].set_weight('bold')
    # labels[-1].set_weight('bold')
    if title:
        ax.set_title(title)
    return ax


def spikesplot_cb(position, cmap='viridis', fig=None):
    # Add colorbar
    if fig is None:
        fig = plt.gcf()

    cax = fig.add_axes(position)
    cb = ColorbarBase(cax, cmap=get_cmap(cmap), spacing='proportional',
                      orientation='horizontal', drawedges=False)
    cb.set_ticks([0, 0.5, 1.0])
    cb.set_ticklabels(['Inferior', '(axial slice)', 'Superior'])
    cb.outline.set_linewidth(0)
    cb.ax.xaxis.set_tick_params(width=0)
    return cax


def confoundplot(tseries, gs_ts, gs_dist=None, name=None, normalize=True,
                 units=None, tr=None, hide_x=True, color='b', nskip=0,
                 cutoff=None, ylims=None):

    # Define TR and number of frames
    notr = False
    if tr is None:
        notr = True
        tr = 1.
    ntsteps = len(tseries)

    # Normalize time series
    tseries = np.array(tseries)
    if normalize:
        tseries /= tr

    # Define nested GridSpec
    gs = mgs.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_ts,
                                     width_ratios=[1, 100], wspace=0.0)

    ax_ts = plt.subplot(gs[1])
    ax_ts.grid(False)
    ax_ts.plot(tseries, color=color)
    ax_ts.set_xlim((0, ntsteps - 1))

    # Set 10 frame markers in X axis
    interval = ntsteps // 10
    xticks = list(range(0, ntsteps)[::interval]) + [ntsteps - 1]
    ax_ts.set_xticks(xticks)

    if not hide_x:
        if notr:
            ax_ts.set_xlabel('time (frame #)')
        else:
            ax_ts.set_xlabel('time (s)')
            labels = tr * np.array(xticks)
            ax_ts.set_xticklabels(['%.02f' % t for t in labels.tolist()])
    else:
        ax_ts.set_xticklabels([])

    no_scale = notr or not normalize
    if not name is None:
        var_label = name
        if not units is None:
            var_label += (' [{}]' if no_scale else ' [{}/s]').format(units)
        ax_ts.set_ylabel(var_label)

    for side in ["top", "right"]:
        ax_ts.spines[side].set_color('none')
        ax_ts.spines[side].set_visible(False)

    if not hide_x:
        ax_ts.spines["bottom"].set_position(('outward', 20))
        ax_ts.xaxis.set_ticks_position('bottom')
    else:
        ax_ts.spines["bottom"].set_color('none')
        ax_ts.spines["bottom"].set_visible(False)

    ax_ts.spines["left"].set_position(('outward', 30))
    ax_ts.yaxis.set_ticks_position('left')

    # Calculate Y limits
    def_ylims = [0.95 * tseries[~np.isnan(tseries)].min(),
                 1.1 * tseries[~np.isnan(tseries)].max()]
    if ylims is not None:
        if ylims[0] is not None:
            def_ylims[0] = min([def_ylims[0], ylims[0]])
        if ylims[1] is not None:
            def_ylims[1] = max([def_ylims[1], ylims[1]])

    ax_ts.set_ylim(def_ylims)
    yticks = sorted(def_ylims)
    ax_ts.set_yticks(yticks)
    ax_ts.set_yticklabels(['%.02f' % y for y in yticks])
    yrange = def_ylims[1] - def_ylims[0]

    # Plot average
    if cutoff is None:
        cutoff = []

    cutoff.insert(0, tseries[~np.isnan(tseries)].mean())

    for i, thr in enumerate(cutoff):
        ax_ts.plot((0, ntsteps - 1), [thr] * 2,
                   linewidth=.75,
                   linestyle='-' if i == 0 else ':',
                   color=color if i == 0 else 'k')

        if i == 0:
            mean_label = r'$\mu$=%.3f%s' % (thr, units if units is not None else '')
            ax_ts.annotate(
                mean_label, xy=(ntsteps - 1, thr), xytext=(11, 0),
                textcoords='offset points', va='center', color='w', size=10,
                bbox=dict(boxstyle='round', fc=color, ec='none', color='none', lw=0),
                arrowprops=dict(
                    arrowstyle='wedge,tail_width=0.8', lw=0, patchA=None, patchB=None,
                    fc=color, ec='none', relpos=(0.01, 0.5)))
        else:
            y_off = [0.0, 0.0]
            for pth in cutoff[:i]:
                inc = abs(thr - pth)
                if inc < yrange:
                    factor = (- (inc / yrange) + 1) ** 2
                    if (thr - pth) < 0.0:
                        y_off[0] -= factor * 20
                    else:
                        y_off[1] += factor * 20

            offset = y_off[0] if abs(y_off[0]) > y_off[1] else y_off[1]

            a_label = '%.2f%s' % (thr, units if units is not None else '')
            ax_ts.annotate(
                a_label, xy=(ntsteps - 1, thr), xytext=(11, offset),
                textcoords='offset points', va='center',
                color='w', size=10,
                bbox=dict(boxstyle='round', fc='dimgray', ec='none', color='none', lw=0),
                arrowprops=dict(
                    arrowstyle='wedge,tail_width=.9', lw=0, patchA=None, patchB=None,
                    fc='dimgray', ec='none', relpos=(.1, .5)))

    if not gs_dist is None:
        ax_dist = plt.subplot(gs_dist)
        sns.displot(tseries, vertical=True, ax=ax_dist)
        ax_dist.set_xlabel('Timesteps')
        ax_dist.set_ylim(ax_ts.get_ylim())
        ax_dist.set_yticklabels([])

        return [ax_ts, ax_dist], gs
    else:
        return ax_ts, gs
