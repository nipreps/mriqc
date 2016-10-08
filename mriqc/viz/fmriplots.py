#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
from __future__ import print_function, division, absolute_import, unicode_literals

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib import gridspec as mgs
from matplotlib.colors import Normalize
from matplotlib.colorbar import ColorbarBase

def fmricarpetplot(func_data, segmentation, outer_gs, tr=None, nskip=4):
    """
    Plot "the plot"
    """
    from nilearn.signal import clean

    # Define TR and number of frames
    notr=False
    if tr is None:
        notr = True
        tr = 1.
    ntsteps = func_data.shape[-1]

    data = func_data[segmentation>1, :].reshape(-1, ntsteps)
    # Detrend data
    detrended = clean(data[:, nskip:].T, t_r=tr).T

    # Order following segmentation labels
    seg = segmentation[segmentation>1].reshape(-1)
    order = np.argsort(seg)

    # Define nested GridSpec
    gs = mgs.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer_gs,
                                     width_ratios=[1, 100], wspace=0.0)

    # Segmentation colorbar
    ax0 = plt.subplot(gs[0])
    ax0.set_yticks([])
    ax0.set_xticks([])
    ax0.imshow(seg[order, np.newaxis], interpolation='nearest', aspect='auto')
    ax0.grid(False)
    ax0.set_ylabel('voxels')

    # Carpet plot
    ax1 = plt.subplot(gs[1])
    theplot = ax1.imshow(detrended[order, :], interpolation='nearest',
                        aspect='auto', cmap='gray', vmin=-2, vmax=2)

    ax1.grid(False)
    ax1.set_yticks([])
    ax1.set_yticklabels([])

    # Set 10 frame markers in X axis
    interval = int(detrended.shape[-1] + 1) // 10
    xticks = list(range(0, detrended.shape[-1])[::interval]) + [detrended.shape[-1]-1]
    ax1.set_xticks(xticks)

    if notr:
        ax1.set_xlabel('time (frame #)')
    else:
        ax1.set_xlabel('time (s)')
        labels = tr * (np.array(xticks) + nskip)
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

def spikesplot(tsz_file, outer_gs=None, tr=None, spike_thresh=6., title='Spike plot',
               ax=None, cmap='viridis', hide_x=True, nskip=4):
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
    notr=False
    if tr is None:
        notr = True
        tr = 1.

    # Load timeseries, zscored slice-wise
    ts_z = np.loadtxt(tsz_file)
    nslices = ts_z.shape[0]
    ntsteps = ts_z.shape[1]

    # Load a colormap
    my_cmap = get_cmap(cmap)
    norm = Normalize(vmin=0, vmax=float(nslices - 1))
    colors = [my_cmap(norm(sl)) for sl in range(nslices)]

    # Plot one line per axial slice timeseries
    for sl in range(nslices):
        ax.plot(ts_z[sl,nskip:].data, color=colors[sl], lw=3)

    # Handle X, Y axes
    ax.grid(False)

    # Handle X axis
    last = ntsteps - nskip - 1
    ax.set_xlim(nskip, last)
    xticks = list(range(nskip, last)[::20]) + [last] if not hide_x else []
    ax.set_xticks(xticks)

    if not hide_x:
        if tr is None:
            ax.set_xlabel('time (frame #)')
        else:
            ax.set_xlabel('time (s)')
            ax.set_xticklabels(['%.02f' % t for t in (tr * np.array(xticks)).tolist()])

    # Handle Y axis
    ax.set_ylabel('z-score of axial slice')
    ax.set_ylim((-(np.abs(ts_z).max()) * 1.05, (np.abs(ts_z).max()) * 1.05))
    zs_max = np.abs(ts_z).max()
    ytick_vals = np.arange(0.0, zs_max, 2.0)
    yticks = list(reversed((-1.0 * ytick_vals[ytick_vals > 0]).tolist())) + ytick_vals.tolist()
    yticks.insert(0, ts_z.min())
    yticks += [ts_z.max()]
    ax.set_yticks(yticks)
    ax.set_yticklabels(['%.02f' % y for y in yticks])
    for val in ytick_vals:
        ax.plot((0,ntsteps),(-val,-val),'k:', alpha=.2)
        ax.plot((0,ntsteps),(val,val),'k:', alpha=.2)

    # Plot maximum and minimum horizontal lines
    ax.plot((0,ntsteps), (yticks[0],yticks[0]),'k:')
    ax.plot((0,ntsteps), (yticks[-1],yticks[-1]),'k:')

    # Plot spike threshold
    if zs_max < spike_thresh:
        ax.plot((0,ts_z.shape[1]),(-spike_thresh,-spike_thresh),'k:')
        ax.plot((0,ts_z.shape[1]),(spike_thresh,spike_thresh),'k:')

    for side in ["top", "right"]:
        ax.spines[side].set_color('none')
        ax.spines[side].set_visible(False)

    if not hide_x:
        ax.spines["bottom"].set_position(('outward', 20))
        ax.xaxis.set_ticks_position('bottom')
    else:
        ax.spines["bottom"].set_color('none')
        ax.spines["bottom"].set_visible(False)

    ax.spines["left"].set_position(('outward', 20))
    ax.yaxis.set_ticks_position('left')

    labels = [label for label in ax.yaxis.get_ticklabels()]
    labels[0].set_weight('bold')
    labels[-1].set_weight('bold')

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
