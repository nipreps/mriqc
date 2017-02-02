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

import math
import os.path as op
import numpy as np
import nibabel as nb
import pandas as pd

from nilearn.plotting import plot_anat, plot_roi

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.backends.backend_pdf import FigureCanvasPdf as FigureCanvas
import seaborn as sns

from builtins import zip, range, str, bytes
from .svg import combine_svg, svg2str

DEFAULT_DPI = 300
DINA4_LANDSCAPE = (11.69, 8.27)
DINA4_PORTRAIT = (8.27, 11.69)

def plot_slice_tern(dslice, prev=None, post=None,
                    spacing=None, cmap='Greys_r', label=None, ax=None,
                    vmax=None, vmin=None, annotate=False):
    from matplotlib.cm import get_cmap

    if isinstance(cmap, (str, bytes)):
        cmap = get_cmap(cmap)

    est_vmin, est_vmax = _get_limits(dslice)
    if not vmin:
        vmin = est_vmin
    if not vmax:
        vmax = est_vmax

    if ax is None:
        ax = plt.gca()

    if spacing is None:
        spacing = [1.0, 1.0]
    else:
        spacing = [spacing[1], spacing[0]]

    phys_sp = np.array(spacing) * dslice.shape

    if prev is None:
        prev = np.ones_like(dslice)
    if post is None:
        post = np.ones_like(dslice)

    combined = np.swapaxes(np.vstack((prev, dslice, post)), 0, 1)
    ax.imshow(combined, vmin=vmin, vmax=vmax, cmap=cmap,
              interpolation='nearest', origin='lower',
              extent=[0, phys_sp[1] * 3, 0, phys_sp[0]])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(False)

    if label is not None:
        ax.text(.5, .05, label,
                transform=ax.transAxes,
                horizontalalignment='center',
                verticalalignment='top',
                size=24,
                bbox=dict(boxstyle="square,pad=0", ec='k', fc='k'),
                color='w')


def plot_spikes(in_file, in_fft, spikes_list, cols=3,
                labelfmt='t={0:.3f}s (z={1:d})',
                out_file=None):
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    nii = nb.as_closest_canonical(nb.load(in_file))
    fft = nb.load(in_fft).get_data()


    data = nii.get_data()
    zooms = nii.header.get_zooms()[:2]
    ntpoints = data.shape[-1]

    if len(spikes_list) > cols * 7:
        cols += 1


    nspikes = len(spikes_list)
    rows = 1
    if nspikes > cols:
        rows = math.ceil(nspikes / cols)

    fig = plt.figure(figsize=(7 * cols, 5 * rows))

    for i, (t, z) in enumerate(spikes_list):
        prev = None
        pvft = None
        if t > 0:
            prev = data[..., z, t - 1]
            pvft = fft[..., z, t - 1]

        post = None
        psft = None
        if t < (ntpoints - 1):
            post = data[..., z, t + 1]
            psft = fft[..., z, t + 1]


        ax1 = fig.add_subplot(rows, cols, i + 1)
        divider = make_axes_locatable(ax1)
        ax2 = divider.new_vertical(size="100%", pad=0.1)
        fig.add_axes(ax2)

        plot_slice_tern(data[..., z, t], prev=prev, post=post, spacing=zooms,
                        ax=ax2,
                        label=labelfmt.format(t, z))

        plot_slice_tern(fft[..., z, t], prev=pvft, post=psft, vmin=-5, vmax=5, cmap='coolwarm',
                        ax=ax1)

    plt.tight_layout()
    if out_file is None:
        fname, ext = op.splitext(op.basename(in_file))
        if ext == '.gz':
            fname, _ = op.splitext(fname)
        out_file = op.abspath('%s.svg' % fname)

    fig.savefig(out_file, format='svg', dpi=300, bbox_inches='tight')
    return out_file


def plot_mosaic(img, out_file, ncols=6, title=None, overlay_mask=None,
                threshold=None, bbox_mask_file=None, only_plot_noise=False,
                vmin=None, vmax=None, cmap='Greys_r', plot_sagittal=True):
    from builtins import bytes, str  # pylint: disable=W0622
    from matplotlib import cm
    from nilearn._utils import check_niimg_3d
    from nilearn._utils.niimg import _safe_get_data
    from nilearn._utils.compat import get_affine as _get_affine
    from nilearn._utils.extmath import fast_abs_percentile
    from nilearn._utils.numpy_conversions import as_ndarray
    from nilearn.image import new_img_like

    if isinstance(cmap, (str, bytes)):
        cmap = cm.get_cmap(cmap)


    # This code is copied from nilearn
    if img is not False and img is not None:
        img = check_niimg_3d(img, dtype='auto')
        data = _safe_get_data(img)
        affine = _get_affine(img)

        if np.isnan(np.sum(data)):
            data = np.nan_to_num(data)

        # Deal with automatic settings of plot parameters
        if threshold == 'auto':
            # Threshold epsilon below a percentile value, to be sure that some
            # voxels pass the threshold
            threshold = fast_abs_percentile(data) - 1e-5

        img = new_img_like(img, as_ndarray(data), affine)
    else:
        raise RuntimeError('input image should be a path or a Nifti object')


    start_idx = [0, 0, 0]
    end_idx = (np.array(img.get_shape()) - np.ones(3)).astype(np.uint8).tolist()
    if bbox_mask_file:
        bbox_mask_file = check_niimg_3d(bbox_mask_file, dtype='auto')
        bbox_data = _safe_get_data(bbox_mask_file)
        bbox = np.argwhere(bbox_data)
        start_idx = bbox.min(0)
        end_idx = bbox.max(0) + 1
    elif end_idx[2] > 70:
        start_idx[2] += 15
        end_idx[2] -= 15


    # Zoom in
    data = data[start_idx[0]:end_idx[0],
                start_idx[1]:end_idx[1],
                start_idx[2]:end_idx[2]]

    # Move center of coordinates
    if sum(start_idx) > 0:
        affine[:3, 3] += affine[:3, :3].dot(start_idx)

    img = new_img_like(img, as_ndarray(data), affine)

    z_cuts = np.array(list(range(data.shape[2])))

    while len(z_cuts) > 36:
        # Discard one every two slices
        z_cuts = z_cuts[::2]

    # Discard first N volumes to make it multiple of ncols
    z_cuts = z_cuts[len(z_cuts) % ncols:]
    z_grouped_cuts = [z_cuts[i:i + ncols] for i in range(0, len(z_cuts), ncols)]

    overlay_data = None
    if overlay_mask:
        overlay_mask = check_niimg_3d(overlay_mask, dtype='auto')
        overlay_data = _safe_get_data(overlay_mask)


    est_vmin, est_vmax = _get_limits(
        data, only_plot_noise=only_plot_noise)
    if not vmin:
        vmin = est_vmin
    if not vmax:
        vmax = est_vmax

    svg_rows = []
    for row, row_cuts in enumerate(z_grouped_cuts):
        plot_kwargs = {
            'title': title if row == 0 else None,
            'display_mode': 'z',
            'cut_coords': [affine.dot([0, 0, r, 1])[2] for r in row_cuts],
            'vmax': vmax,
            'vmin': vmin,
            'cmap': cmap
        }

        if overlay_data is None:
            display = plot_anat(img, **plot_kwargs)
        else:
            display = plot_roi(overlay_data, bg_img=img,
                               **plot_kwargs)

        svg_rows.append(svg2str(display))
        display.close()
        display = None


    if plot_sagittal:
        x_sp = data.shape[0] // (ncols + 1)
        x_vox = list(range(x_sp, data.shape[0], x_sp))
        x_coords = [affine.dot([x, 0, 0, 1])[0] for x in x_vox[:-1]]

        plot_kwargs = {
            'display_mode': 'x',
            'cut_coords': x_coords,
            'vmax': vmax,
            'vmin': vmin,
            'cmap': cmap
        }

        if overlay_data is None:
            display = plot_anat(img, **plot_kwargs)
        else:
            display = plot_roi(overlay_data, bg_img=img,
                               **plot_kwargs)

        svg_rows.append(svg2str(display))
        display.close()
        display = None

    fig = combine_svg(svg_rows)
    fig.save(out_file)
    return out_file


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


def plot_mosaic_helper(in_file, out_file=None, bbox_mask_file=None, title=None,
                       plot_sagittal=True, only_plot_noise=False, cmap='Greys_r'):

    if out_file is None:
        fname, ext = op.splitext(op.basename(in_file))
        if ext == ".gz":
            fname, _ = op.splitext(fname)
        out_file = fname + '_mosaic.svg'

    out_file = op.abspath(out_file)
    plot_mosaic(
        in_file, out_file, bbox_mask_file=bbox_mask_file, title=title,
        only_plot_noise=only_plot_noise, cmap=cmap, plot_sagittal=plot_sagittal
    )
    return out_file


def _get_limits(nifti_file, only_plot_noise=False):
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
