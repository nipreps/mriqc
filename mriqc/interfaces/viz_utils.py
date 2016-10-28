#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# @Author: oesteban
# @Date:   2016-01-05 11:32:01
# @Email:  code@oscaresteban.es
# @Last modified by:   oesteban
# @Last Modified time: 2016-10-27 16:41:53
""" Visualization utilities """
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from __future__ import unicode_literals
from builtins import str
from builtins import zip
from builtins import range

import math
import time
import os.path as op
import numpy as np
import nibabel as nb
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.backends.backend_pdf import FigureCanvasPdf as FigureCanvas
import seaborn as sns

DEFAULT_DPI = 300
DINA4_LANDSCAPE = (11.69, 8.27)
DINA4_PORTRAIT = (8.27, 11.69)


def plot_measures(df, measures, ncols=4, title='Group level report',
                  subject=None, figsize=DINA4_PORTRAIT):
    import matplotlib.gridspec as gridspec
    nmeasures = len(measures)
    nrows = nmeasures // ncols
    if nmeasures % ncols > 0:
        nrows += 1

    fig = plt.figure(figsize=figsize)
    gsp = gridspec.GridSpec(nrows, ncols)

    axes = []

    for i, mname in enumerate(measures):
        axes.append(plt.subplot(gsp[i]))
        axes[-1].set_xlabel(mname)
        sns.distplot(
            df[[mname]], ax=axes[-1], color="b", rug=True,  norm_hist=True)

        # labels = np.array(axes[-1].get_xticklabels())
        # labels[2:-2] = ''
        axes[-1].set_xticklabels([])
        plt.ticklabel_format(style='sci', axis='y', scilimits=(-1, 1))

        if subject is not None:
            subid = subject
            try:
                subid = int(subid)
            except ValueError:
                pass

            subdf = df.loc[df['subject_id'] == subid]
            sessions = np.atleast_1d(subdf[['session_id']]).reshape(-1).tolist()

            for ss in sessions:
                sesdf = subdf.loc[subdf['session_id'] == ss]
                scans = np.atleast_1d(sesdf[['run_id']]).reshape(-1).tolist()

                for sc in scans:
                    scndf = subdf.loc[sesdf['run_id'] == sc]
                    plot_vline(
                        scndf.iloc[0][mname], '_'.join([ss, sc]), axes[-1])

    fig.suptitle(title)
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    plt.subplots_adjust(top=0.85)
    return fig


def plot_all(df, groups, subject=None, figsize=(DINA4_LANDSCAPE[0], 5),
             strip_nsubj=10, title='Summary report'):
    import matplotlib.gridspec as gridspec
    # colnames = [v for gnames in groups for v in gnames]
    lengs = [len(el) for el in groups]
    # ncols = np.sum(lengs)

    fig = plt.figure(figsize=figsize)
    gsp = gridspec.GridSpec(1, len(groups), width_ratios=lengs)

    subjects = sorted(pd.unique(df.subject_id.ravel()))
    nsubj = len(subjects)
    subid = subject
    if subid is not None:
        try:
            subid = int(subid)
        except ValueError:
            pass

    axes = []
    for i, snames in enumerate(groups):
        if len(snames) == 0:
            continue

        axes.append(plt.subplot(gsp[i]))

        if nsubj > strip_nsubj:
            pal = sns.color_palette("hls", len(snames))
            sns.violinplot(data=df[snames], ax=axes[-1], linewidth=.8, palette=pal)
        else:
            stdf = df.copy()
            if subid is not None:
                stdf = stdf.loc[stdf['subject_id'] != subid]
            sns.stripplot(data=stdf[snames], ax=axes[-1], jitter=0.25)

        axes[-1].set_xticklabels(
            [el.get_text() for el in axes[-1].get_xticklabels()],
            rotation='vertical')
        plt.ticklabel_format(style='sci', axis='y', scilimits=(-1, 1))
        # df[snames].plot(kind='box', ax=axes[-1])

        # If we know the subject, place a star for each scan
        if subject is not None:
            subdf = df.loc[df['subject_id'] == subid]
            scans = sorted(pd.unique(subdf.run_id.ravel()))
            nstars = len(scans)
            if nstars == 0:
                continue

            for j, sname in enumerate(snames):
                vals = []
                for _, scid in enumerate(scans):
                    val = subdf.loc[df.run_id == scid, [sname]].iloc[0, 0]
                    vals.append(val)

                if len(vals) != nstars:
                    continue

                pos = [j]
                if nstars > 1:
                    pos = np.linspace(j-0.3, j+0.3, num=nstars)

                axes[-1].plot(
                    pos, vals, ms=9, mew=.8, linestyle='None',
                    color='w', marker='*', markeredgecolor='k',
                    zorder=10)

    fig.suptitle(title)
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    plt.subplots_adjust(top=0.85)
    return fig


def plot_mosaic(nifti_file, title=None, overlay_mask=None,
                fig=None, bbox_mask_file=None, only_plot_noise=False,
                figsize=DINA4_LANDSCAPE):
    from six import string_types
    from pylab import cm

    if isinstance(nifti_file, string_types):
        nii = nb.as_closest_canonical(nb.load(nifti_file))
        mean_data = nii.get_data()
    else:
        mean_data = nifti_file

    if bbox_mask_file:
        bbox_data = nb.as_closest_canonical(nb.load(bbox_mask_file)).get_data()
        B = np.argwhere(bbox_data)
        (ystart, xstart, zstart), (ystop, xstop, zstop) = B.min(0), B.max(
            0) + 1
        mean_data = mean_data[ystart:ystop, xstart:xstop, zstart:zstop]

    z_vals = np.array(list(range(0, mean_data.shape[2])))
    # Reduce the number of slices shown
    if mean_data.shape[2] > 70:
        rem = 15
        # Crop inferior and posterior
        if not bbox_mask_file:
            mean_data = mean_data[..., rem:-rem]
            z_vals = z_vals[rem:-rem]
        else:
            mean_data = mean_data[..., 2 * rem:]
            z_vals = z_vals[2 * rem:]

    if mean_data.shape[2] > 70:
        # Discard one every two slices
        mean_data = mean_data[..., ::2]
        z_vals = z_vals[::2]

    n_images = mean_data.shape[2]
    row, col = _calc_rows_columns((figsize[0] / figsize[1]), n_images)

    if overlay_mask:
        overlay_data = nb.as_closest_canonical(
            nb.load(overlay_mask)).get_data()

    # create figures
    if fig is None:
        fig = plt.Figure(figsize=figsize)

    FigureCanvas(fig)

    fig.subplots_adjust(top=0.85)
    for image, z_val in enumerate(z_vals):
        ax = fig.add_subplot(row, col, image + 1)
        data_mask = np.logical_not(np.isnan(mean_data))
        if only_plot_noise:
            data_mask = np.logical_and(data_mask, mean_data != 0)
        if overlay_mask:
            ax.set_rasterized(True)

        if only_plot_noise:
            vmin = np.percentile(mean_data[data_mask], 0)
            vmax = np.percentile(mean_data[data_mask], 61)
        else:
            vmin = np.percentile(mean_data[data_mask], 0.5)
            vmax = np.percentile(mean_data[data_mask], 99.5)

        ax.imshow(np.fliplr(mean_data[:, :, image].T), vmin=vmin,
                  vmax=vmax,
                  cmap=cm.Greys_r, interpolation='nearest', origin='lower')

        if overlay_mask:
            cmap = cm.Reds  # @UndefinedVariable
            cmap._init()
            alphas = np.linspace(0, 0.75, cmap.N + 3)
            cmap._lut[:, -1] = alphas
            ax.imshow(np.fliplr(overlay_data[:, :, image].T), vmin=0, vmax=1,
                      cmap=cmap, interpolation='nearest', origin='lower')

        ax.annotate(
            str(z_val), xy=(.99, .99), xycoords='axes fraction',
            fontsize=8, color='white', horizontalalignment='right',
            verticalalignment='top')

        ax.axis('off')

    fig.subplots_adjust(
        left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.01, hspace=0.1)

    if title:
        fig.suptitle(title, fontsize='10')
    fig.subplots_adjust(wspace=0.002, hspace=0.002)
    return fig


def plot_fd(fd_file, fd_radius, mean_fd_dist=None, figsize=DINA4_LANDSCAPE):

    fd_power = _calc_fd(fd_file, fd_radius)

    fig = plt.Figure(figsize=figsize)
    FigureCanvas(fig)

    if mean_fd_dist:
        grid = GridSpec(2, 4)
    else:
        grid = GridSpec(1, 2, width_ratios=[3, 1])
        grid.update(hspace=1.0, right=0.95, left=0.1, bottom=0.2)

    ax = fig.add_subplot(grid[0, :-1])
    ax.plot(fd_power)
    ax.set_xlim((0, len(fd_power)))
    ax.set_ylabel("Frame Displacement [mm]")
    ax.set_xlabel("Frame number")
    ylim = ax.get_ylim()

    ax = fig.add_subplot(grid[0, -1])
    sns.distplot(fd_power, vertical=True, ax=ax)
    ax.set_ylim(ylim)

    if mean_fd_dist:
        ax = fig.add_subplot(grid[1, :])
        sns.distplot(mean_fd_dist, ax=ax)
        ax.set_xlabel("Mean Frame Displacement (over all subjects) [mm]")
        mean_fd = fd_power.mean()
        label = r'$\overline{{\text{{FD}}}}$ = {0:g}'.format(mean_fd)
        plot_vline(mean_fd, label, ax=ax)

    return fig


def plot_dist(
        main_file, mask_file, xlabel, distribution=None, xlabel2=None,
        figsize=DINA4_LANDSCAPE):
    data = _get_values_inside_a_mask(main_file, mask_file)

    fig = plt.Figure(figsize=figsize)
    FigureCanvas(fig)

    gsp = GridSpec(2, 1)
    ax = fig.add_subplot(gsp[0, 0])
    sns.distplot(data.astype(np.double), kde=False, bins=100, ax=ax)
    ax.set_xlabel(xlabel)

    ax = fig.add_subplot(gsp[1, 0])
    sns.distplot(np.array(distribution).astype(np.double), ax=ax)
    cur_val = np.median(data)
    label = "{0!g}".format(cur_val)
    plot_vline(cur_val, label, ax=ax)
    ax.set_xlabel(xlabel2)

    return fig


def plot_vline(cur_val, label, ax):
    ax.axvline(cur_val)
    ylim = ax.get_ylim()
    vloc = (ylim[0] + ylim[1]) / 2.0
    xlim = ax.get_xlim()
    pad = (xlim[0] + xlim[1]) / 100.0
    ax.text(cur_val - pad, vloc, label, color="blue", rotation=90,
            verticalalignment='center', horizontalalignment='right')


def _calc_rows_columns(ratio, n_images):
    rows = 1
    for _ in range(100):
        columns = math.floor(ratio * rows)
        total = rows * columns
        if total > n_images:
            break
        rows += 1
    return rows, columns


def _calc_fd(fd_file, fd_radius):
    from math import pi
    lines = open(fd_file, 'r').readlines()
    rows = [[float(x) for x in line.split()] for line in lines]
    cols = np.array([list(col) for col in zip(*rows)])

    translations = np.transpose(np.abs(np.diff(cols[0:3, :])))
    rotations = np.transpose(np.abs(np.diff(cols[3:6, :])))

    fd_power = np.sum(translations, axis=1) + \
        (fd_radius * pi / 180) * np.sum(rotations, axis=1)

    # FD is zero for the first time point
    fd_power = np.insert(fd_power, 0, 0)

    return fd_power


def _get_mean_fd_distribution(fd_files, fd_radius):
    mean_fds = []
    max_fds = []
    for fd_file in fd_files:
        fd_power = _calc_fd(fd_file, fd_radius)
        mean_fds.append(fd_power.mean())
        max_fds.append(fd_power.max())

    return mean_fds, max_fds


def _get_values_inside_a_mask(main_file, mask_file):
    main_nii = nb.load(main_file)
    main_data = main_nii.get_data()
    nan_mask = np.logical_not(np.isnan(main_data))
    mask = nb.load(mask_file).get_data() > 0

    data = main_data[np.logical_and(nan_mask, mask)]
    return data


def plot_segmentation(anat_file, segmentation, out_file,
                      **kwargs):
    from nilearn.plotting import plot_anat

    vmax = None
    if kwargs.get('saturate', False):
        import nibabel as nb
        import numpy as np
        vmax = np.percentile(nb.load(anat_file).get_data().reshape(-1),
                             70)

    disp = plot_anat(
        anat_file,
        display_mode=kwargs.get('display_mode', 'ortho'),
        cut_coords=kwargs.get('cut_coords', 8),
        title=kwargs.get('title'),
        vmax=vmax)
    disp.add_contours(
        segmentation,
        levels=kwargs.get('levels', [1]),
        colors=kwargs.get('colors', 'r'))
    disp.savefig(out_file)
    disp.close()
    disp = None
    return out_file

def plot_bg_dist(in_file):
    import os.path as op
    import numpy as np
    import json
    from io import open # pylint: disable=W0622
    import matplotlib.pyplot as plt
    # rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
    # rc('text', usetex=True)

    with open(in_file, 'r') as jsonf:
        data = json.load(jsonf)

    # Write out figure of the fitting
    out_file = op.abspath('background_fit.svg')
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    fig.suptitle('Noise distribution on the air mask, and fitted chi distribution')
    ax1.set_xlabel('Intensity')
    ax1.set_ylabel('Frequency')

    width = (data['x'][1] - data['x'][0])
    left = [v - 0.5 * width for v in data['x']]

    ymax = np.max([np.array(data['y']).max(), np.array(data['y_hat']).max()])
    ax1.set_ylim((0.0, 1.10 * ymax))

    ax1.bar(left, data['y'], width)
    ax1.plot(left, data['y_hat'], 'k--', linewidth=1.2)
    ax1.plot((data['x_cutoff'], data['x_cutoff']), ax1.get_ylim(), 'k--')

    fig.savefig(out_file, format='svg', dpi=300)
    plt.close()
    return out_file


def plot_mosaic_helper(in_file, subject_id, session_id,
                       run_id, out_name, bbox_mask_file=None,
                       title=None,
                       only_plot_noise=False):
    if title is not None:
        title = title.format(**{"session_id": session_id,
                              "run_id": run_id})
    fig = plot_mosaic(in_file, bbox_mask_file=bbox_mask_file, title=title,
                      only_plot_noise=only_plot_noise)
    fig.savefig(out_name, format=out_name.split('.')[-1], dpi=300)
    fig.clf()
    fig = None
    return op.abspath(out_name)

def combine_svg_verbose(
        in_brainmask,
        in_segmentation,
        in_artmask,
        in_headmask,
        in_airmask,
        in_bgplot):
    import os.path as op
    import svgutils.transform as svgt
    import svgutils.compose as svgc
    import numpy as np

    hspace = 10
    wspace = 10
    #create new SVG figure
    in_mosaics = [in_brainmask,
                  in_segmentation,
                  in_artmask,
                  in_headmask,
                  in_airmask]
    figs = [svgt.fromfile(f) for f in in_mosaics]

    roots = [f.getroot() for f in figs]
    nfigs = len(figs)

    sizes = [(int(f.width[:-2]), int(f.height[:-2])) for f in figs]
    maxsize = np.max(sizes, axis=0)
    minsize = np.min(sizes, axis=0)

    bgfile = svgt.fromfile(in_bgplot)
    bgscale = (maxsize[1] * 2 + hspace)/int(bgfile.height[:-2])
    bgsize = (int(bgfile.width[:-2]), int(bgfile.height[:-2]))
    bgfileroot = bgfile.getroot()

    totalsize = (minsize[0] + hspace + int(bgsize[0] * bgscale),
                 nfigs * maxsize[1] + (nfigs - 1) * hspace)
    fig = svgt.SVGFigure(svgc.Unit(totalsize[0]).to('cm'),
                         svgc.Unit(totalsize[1]).to('cm'))

    yoffset = 0
    for i, r in enumerate(roots):
        xoffset = 0
        if sizes[i][0] == maxsize[0]:
            xoffset = int(0.5 * (totalsize[0] - sizes[i][0]))
        r.moveto(xoffset, yoffset)
        yoffset += maxsize[1] + hspace

    bgfileroot.moveto(minsize[0] + wspace, 3 * (maxsize[1] + hspace), scale=bgscale)

    fig.append(roots + [bgfileroot])
    out_file = op.abspath('fig_final.svg')
    fig.save(out_file)
    return out_file
