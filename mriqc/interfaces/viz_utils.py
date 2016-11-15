#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# @Author: oesteban
# @Date:   2016-01-05 11:32:01
# @Email:  code@oscaresteban.es
# @Last modified by:   oesteban
""" Visualization utilities """
from __future__ import print_function, division, absolute_import, unicode_literals
from builtins import zip, range

import math
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
            df[[mname]], ax=axes[-1], color="b", rug=True, norm_hist=True)

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


def get_limits(nifti_file, only_plot_noise=False):
    from builtins import bytes, str   # pylint: disable=W0622

    if isinstance(nifti_file, (str, bytes)):
        nii = nb.as_closest_canonical(nb.load(nifti_file))
        data = nii.get_data()
    else:
        data = nifti_file

    data_mask = np.logical_not(np.isnan(data))

    if only_plot_noise:
        data_mask = np.logical_and(data_mask, data != 0)
        vmin = np.percentile(data[data_mask], 0)
        vmax = np.percentile(data[data_mask], 61)
    else:
        vmin = np.percentile(data[data_mask], 0.5)
        vmax = np.percentile(data[data_mask], 99.5)

    return vmin, vmax


def plot_mosaic(nifti_file, title=None, overlay_mask=None,
                fig=None, bbox_mask_file=None, only_plot_noise=False,
                vmin=None, vmax=None, figsize=DINA4_LANDSCAPE,
                cmap='Greys_r', plot_sagittal=True, labels=None):
    from builtins import bytes, str  # pylint: disable=W0622
    from matplotlib import cm

    if isinstance(cmap, (str, bytes)):
        cmap = cm.get_cmap(cmap)

    if isinstance(nifti_file, (str, bytes)):
        nii = nb.as_closest_canonical(nb.load(nifti_file))
        mean_data = nii.get_data()
        mean_data = mean_data[::-1, ...]
    else:
        mean_data = nifti_file

    if bbox_mask_file:
        bbox_data = nb.as_closest_canonical(
            nb.load(bbox_mask_file)).get_data()
        bbox_data = bbox_data[::-1, ...]
        B = np.argwhere(bbox_data)
        (ystart, xstart, zstart), (ystop, xstop, zstop) = B.min(0), B.max(
            0) + 1
        mean_data = mean_data[ystart:ystop, xstart:xstop, zstart:zstop]

    z_vals = np.array(list(range(0, mean_data.shape[2])))

    if labels is None:
        # Reduce the number of slices shown
        if len(z_vals) > 70:
            rem = 15
            # Crop inferior and posterior
            if not bbox_mask_file:
                # mean_data = mean_data[..., rem:-rem]
                z_vals = z_vals[rem:-rem]
            else:
                # mean_data = mean_data[..., 2 * rem:]
                z_vals = z_vals[2 * rem:]

        while len(z_vals) > 70:
            # Discard one every two slices
            # mean_data = mean_data[..., ::2]
            z_vals = z_vals[::2]

        labels = ['%d' % z for z in z_vals]

    n_images = len(z_vals)
    row, col = _calc_rows_columns((figsize[0] / figsize[1]), n_images)

    end = "pre"
    z_vals = list(z_vals)
    while (row - 1) * col > len(z_vals) and (
            z_vals[0] != 0 or z_vals[-1] != mean_data.shape[2] - 1):
        if end == "pre":
            if z_vals[0] != 0:
                z_vals = [z_vals[0] - 1] + z_vals
            end = "post"
        else:
            if z_vals[-1] != mean_data.shape[2] - 1:
                z_vals = z_vals + [z_vals[-1] + 1]
            end = "pre"
        if (row - 1) * col < len(z_vals):
            break

    if overlay_mask:
        overlay_data = nb.as_closest_canonical(
            nb.load(overlay_mask)).get_data()

    # create figures
    if fig is None:
        fig = plt.Figure(figsize=figsize)

    FigureCanvas(fig)

    est_vmin, est_vmax = get_limits(mean_data,
                                    only_plot_noise=only_plot_noise)
    if not vmin:
        vmin = est_vmin
    if not vmax:
        vmax = est_vmax

    fig.subplots_adjust(top=0.85)
    for image, (z_val, z_label) in enumerate(zip(z_vals, labels)):
        ax = fig.add_subplot(row, col, image + 1)
        if overlay_mask:
            ax.set_rasterized(True)
        ax.imshow(np.fliplr(mean_data[:, :, z_val].T), vmin=vmin,
                  vmax=vmax,
                  cmap=cmap, interpolation='nearest', origin='lower')

        if overlay_mask:
            cmap = cm.Reds  # @UndefinedVariable
            cmap._init()
            alphas = np.linspace(0, 0.75, cmap.N + 3)
            cmap._lut[:, -1] = alphas
            ax.imshow(np.fliplr(overlay_data[:, :, z_val].T), vmin=0,
                      vmax=1,
                      cmap=cmap, interpolation='nearest', origin='lower')

        ax.annotate(
            z_label, xy=(.99, .99), xycoords='axes fraction',
            fontsize=8, color='k', backgroundcolor='white', horizontalalignment='right',
            verticalalignment='top')

        ax.axis('off')

    if plot_sagittal:
        start = int(mean_data.shape[0] / 5)
        stop = mean_data.shape[0] - start
        step = int((stop - start) / (col))
        x_vals = range(start, stop, step)
        x_vals = np.array(x_vals[:col])
        x_vals += int((stop - x_vals[-1]) / 2)
        for image, x_val in enumerate(x_vals):
            ax = fig.add_subplot(row, col, image + (row - 1) * col + 1)
            ax.imshow(mean_data[x_val, :, :].T, vmin=vmin,
                      vmax=vmax,
                      cmap=cmap, interpolation='nearest', origin='lower')
            ax.annotate(
                '%d' % x_val, xy=(.99, .99), xycoords='axes fraction',
                fontsize=8, color='k', backgroundcolor='white',
                horizontalalignment='right', verticalalignment='top')
            ax.axis('off')

    fig.subplots_adjust(
        left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.01,
        hspace=0.1)

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
    rows = 2
    for _ in range(100):
        columns = math.floor(ratio * rows)
        total = (rows - 1) * columns
        if total > n_images:
            rows = np.ceil(n_images / columns) + 1
            break
        rows += 1
    return int(rows), int(columns)


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
    import nibabel as nb
    import numpy as np
    from nilearn.plotting import plot_anat

    vmax = kwargs.get('vmax')
    vmin = kwargs.get('vmin')

    if kwargs.get('saturate', False):
        vmax = np.percentile(nb.load(anat_file).get_data().reshape(-1), 70)

    if vmax is None and vmin is None:

        vmin = np.percentile(nb.load(anat_file).get_data().reshape(-1), 10)
        vmax = np.percentile(nb.load(anat_file).get_data().reshape(-1), 99)

    disp = plot_anat(
        anat_file,
        display_mode=kwargs.get('display_mode', 'ortho'),
        cut_coords=kwargs.get('cut_coords', 8),
        title=kwargs.get('title'),
        vmax=vmax, vmin=vmin)
    disp.add_contours(
        segmentation,
        levels=kwargs.get('levels', [1]),
        colors=kwargs.get('colors', 'r'))
    disp.savefig(out_file)
    disp.close()
    disp = None
    return out_file


def plot_bg_dist(in_file):
    import os.path as op  # pylint: disable=W0621
    import numpy as np
    import json
    from io import open # pylint: disable=W0622
    import matplotlib.pyplot as plt
    # rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
    # rc('text', usetex=True)

    # Write out figure of the fitting
    out_file = op.abspath('background_fit.svg')
    try:
        with open(in_file, 'r') as jsonf:
            data = json.load(jsonf)
    except ValueError:
        with open(out_file, 'w') as ofh:
            ofh.write('<p>Background noise fitting could not be plotted.</p>')
        return out_file

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


def plot_mosaic_helper(in_file, subject_id, session_id=None,
                       task_id=None, run_id=None, out_file=None, bbox_mask_file=None,
                       title=None, plot_sagittal=True, labels=None,
                       only_plot_noise=False, cmap='Greys_r'):
    if title is not None:
        title = title.format(**{"session_id": session_id,
                                "task_id": task_id,
                                "run_id": run_id})
    fig = plot_mosaic(in_file, bbox_mask_file=bbox_mask_file, title=title, labels=labels,
                      only_plot_noise=only_plot_noise, cmap=cmap, plot_sagittal=plot_sagittal)
    fig.savefig(out_file, format=out_file.split('.')[-1], dpi=300)
    fig.clf()
    fig = None
    return op.abspath(out_file)

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
