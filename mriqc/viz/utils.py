# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
""" Visualization utilities """

import math
import os.path as op
import numpy as np
import nibabel as nb

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.backends.backend_pdf import FigureCanvasPdf as FigureCanvas
import seaborn as sns

DEFAULT_DPI = 300
DINA4_LANDSCAPE = (11.69, 8.27)
DINA4_PORTRAIT = (8.27, 11.69)


def plot_slice(
    dslice,
    spacing=None,
    cmap="Greys_r",
    label=None,
    ax=None,
    vmax=None,
    vmin=None,
    annotate=False,
):
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

    phys_sp = np.array(spacing) * dslice.shape
    ax.imshow(
        np.swapaxes(dslice, 0, 1),
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        interpolation="nearest",
        origin="lower",
        extent=[0, phys_sp[0], 0, phys_sp[1]],
    )
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(False)
    ax.axis("off")

    bgcolor = cmap(min(vmin, 0.0))
    fgcolor = cmap(vmax)

    if annotate:
        ax.text(
            0.95,
            0.95,
            "R",
            color=fgcolor,
            transform=ax.transAxes,
            horizontalalignment="center",
            verticalalignment="top",
            size=18,
            bbox=dict(boxstyle="square,pad=0", ec=bgcolor, fc=bgcolor),
        )
        ax.text(
            0.05,
            0.95,
            "L",
            color=fgcolor,
            transform=ax.transAxes,
            horizontalalignment="center",
            verticalalignment="top",
            size=18,
            bbox=dict(boxstyle="square,pad=0", ec=bgcolor, fc=bgcolor),
        )

    if label is not None:
        ax.text(
            0.98,
            0.01,
            label,
            color=fgcolor,
            transform=ax.transAxes,
            horizontalalignment="right",
            verticalalignment="bottom",
            size=18,
            bbox=dict(boxstyle="square,pad=0", ec=bgcolor, fc=bgcolor),
        )

    return ax


def plot_slice_tern(
    dslice,
    prev=None,
    post=None,
    spacing=None,
    cmap="Greys_r",
    label=None,
    ax=None,
    vmax=None,
    vmin=None,
):
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
    ax.imshow(
        combined,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        interpolation="nearest",
        origin="lower",
        extent=[0, phys_sp[1] * 3, 0, phys_sp[0]],
    )
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(False)

    if label is not None:
        ax.text(
            0.5,
            0.05,
            label,
            transform=ax.transAxes,
            horizontalalignment="center",
            verticalalignment="top",
            size=14,
            bbox=dict(boxstyle="square,pad=0", ec="k", fc="k"),
            color="w",
        )


def plot_spikes(
    in_file, in_fft, spikes_list, cols=3, labelfmt="t={0:.3f}s (z={1:d})", out_file=None
):
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    nii = nb.as_closest_canonical(nb.load(in_file))
    fft = nb.load(in_fft).get_data()

    data = nii.get_data()
    zooms = nii.header.get_zooms()[:2]
    tstep = nii.header.get_zooms()[-1]
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

        plot_slice_tern(
            data[..., z, t],
            prev=prev,
            post=post,
            spacing=zooms,
            ax=ax2,
            label=labelfmt.format(t * tstep, z),
        )

        plot_slice_tern(
            fft[..., z, t],
            prev=pvft,
            post=psft,
            vmin=-5,
            vmax=5,
            cmap=get_parula(),
            ax=ax1,
        )

    plt.tight_layout()
    if out_file is None:
        fname, ext = op.splitext(op.basename(in_file))
        if ext == ".gz":
            fname, _ = op.splitext(fname)
        out_file = op.abspath("%s.svg" % fname)

    fig.savefig(out_file, format="svg", dpi=300, bbox_inches="tight")
    return out_file


def plot_mosaic(
    img,
    out_file=None,
    ncols=8,
    title=None,
    overlay_mask=None,
    bbox_mask_file=None,
    only_plot_noise=False,
    annotate=True,
    vmin=None,
    vmax=None,
    cmap="Greys_r",
    plot_sagittal=True,
    fig=None,
    zmax=128,
):

    if isinstance(img, (str, bytes)):
        nii = nb.as_closest_canonical(nb.load(img))
        img_data = nii.get_data()
        zooms = nii.header.get_zooms()
    else:
        img_data = img
        zooms = [1.0, 1.0, 1.0]
        out_file = "mosaic.svg"

    # Remove extra dimensions
    img_data = np.squeeze(img_data)

    if img_data.shape[2] > zmax and bbox_mask_file is None:
        lowthres = np.percentile(img_data, 5)
        mask_file = np.ones_like(img_data)
        mask_file[img_data <= lowthres] = 0
        img_data = _bbox(img_data, mask_file)

    if bbox_mask_file is not None:
        bbox_data = nb.as_closest_canonical(nb.load(bbox_mask_file)).get_data()
        img_data = _bbox(img_data, bbox_data)

    z_vals = np.array(list(range(0, img_data.shape[2])))

    # Reduce the number of slices shown
    if len(z_vals) > zmax:
        rem = 15
        # Crop inferior and posterior
        if not bbox_mask_file:
            # img_data = img_data[..., rem:-rem]
            z_vals = z_vals[rem:-rem]
        else:
            # img_data = img_data[..., 2 * rem:]
            z_vals = z_vals[2 * rem:]

    while len(z_vals) > zmax:
        # Discard one every two slices
        # img_data = img_data[..., ::2]
        z_vals = z_vals[::2]

    n_images = len(z_vals)
    nrows = math.ceil(n_images / ncols)
    if plot_sagittal:
        nrows += 1

    if overlay_mask:
        overlay_data = nb.as_closest_canonical(nb.load(overlay_mask)).get_data()

    # create figures
    if fig is None:
        fig = plt.figure(figsize=(22, nrows * 3))

    est_vmin, est_vmax = _get_limits(img_data, only_plot_noise=only_plot_noise)
    if not vmin:
        vmin = est_vmin
    if not vmax:
        vmax = est_vmax

    naxis = 1
    for z_val in z_vals:
        ax = fig.add_subplot(nrows, ncols, naxis)

        if overlay_mask:
            ax.set_rasterized(True)
        plot_slice(
            img_data[:, :, z_val],
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            ax=ax,
            spacing=zooms[:2],
            label="%d" % z_val,
            annotate=annotate,
        )

        if overlay_mask:
            from matplotlib import cm

            msk_cmap = cm.Reds  # @UndefinedVariable
            msk_cmap._init()
            alphas = np.linspace(0, 0.75, msk_cmap.N + 3)
            msk_cmap._lut[:, -1] = alphas
            plot_slice(
                overlay_data[:, :, z_val],
                vmin=0,
                vmax=1,
                cmap=msk_cmap,
                ax=ax,
                spacing=zooms[:2],
            )
        naxis += 1

    if plot_sagittal:
        naxis = ncols * (nrows - 1) + 1

        step = int(img_data.shape[0] / (ncols + 1))
        start = step
        stop = img_data.shape[0] - step

        if step == 0:
            step = 1

        for x_val in list(range(start, stop, step))[:ncols]:
            ax = fig.add_subplot(nrows, ncols, naxis)

            plot_slice(
                img_data[x_val, ...],
                vmin=vmin,
                vmax=vmax,
                cmap=cmap,
                ax=ax,
                label="%d" % x_val,
                spacing=[zooms[0], zooms[2]],
            )
            naxis += 1

    fig.subplots_adjust(
        left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.05, hspace=0.05
    )

    if title:
        fig.suptitle(title, fontsize="10")
    fig.subplots_adjust(wspace=0.002, hspace=0.002)

    if out_file is None:
        fname, ext = op.splitext(op.basename(img))
        if ext == ".gz":
            fname, _ = op.splitext(fname)
        out_file = op.abspath(fname + "_mosaic.svg")

    fig.savefig(out_file, format="svg", dpi=300, bbox_inches="tight")
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
        label = r"$\overline{{\text{{FD}}}}$ = {0:g}".format(mean_fd)
        plot_vline(mean_fd, label, ax=ax)

    return fig


def plot_dist(
    main_file,
    mask_file,
    xlabel,
    distribution=None,
    xlabel2=None,
    figsize=DINA4_LANDSCAPE,
):
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
    ax.text(
        cur_val - pad,
        vloc,
        label,
        color="blue",
        rotation=90,
        verticalalignment="center",
        horizontalalignment="right",
    )


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

    lines = open(fd_file, "r").readlines()
    rows = [[float(x) for x in line.split()] for line in lines]
    cols = np.array([list(col) for col in zip(*rows)])

    translations = np.transpose(np.abs(np.diff(cols[0:3, :])))
    rotations = np.transpose(np.abs(np.diff(cols[3:6, :])))

    fd_power = np.sum(translations, axis=1) + (fd_radius * pi / 180) * np.sum(
        rotations, axis=1
    )

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


def plot_segmentation(anat_file, segmentation, out_file, **kwargs):
    from nilearn.plotting import plot_anat

    vmax = kwargs.get("vmax")
    vmin = kwargs.get("vmin")

    if kwargs.get("saturate", False):
        vmax = np.percentile(nb.load(anat_file).get_data().reshape(-1), 70)

    if vmax is None and vmin is None:

        vmin = np.percentile(nb.load(anat_file).get_data().reshape(-1), 10)
        vmax = np.percentile(nb.load(anat_file).get_data().reshape(-1), 99)

    disp = plot_anat(
        anat_file,
        display_mode=kwargs.get("display_mode", "ortho"),
        cut_coords=kwargs.get("cut_coords", 8),
        title=kwargs.get("title"),
        vmax=vmax,
        vmin=vmin,
    )
    disp.add_contours(
        segmentation, levels=kwargs.get("levels", [1]), colors=kwargs.get("colors", "r")
    )
    disp.savefig(out_file)
    disp.close()
    disp = None
    return out_file


def _get_limits(nifti_file, only_plot_noise=False):
    if isinstance(nifti_file, str):
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


def _bbox(img_data, bbox_data):
    B = np.argwhere(bbox_data)
    (ystart, xstart, zstart), (ystop, xstop, zstop) = B.min(0), B.max(0) + 1
    return img_data[ystart:ystop, xstart:xstop, zstart:zstop]


def get_parula():
    from matplotlib.colors import LinearSegmentedColormap

    cm_data = [
        [0.2081, 0.1663, 0.5292],
        [0.2116238095, 0.1897809524, 0.5776761905],
        [0.212252381, 0.2137714286, 0.6269714286],
        [0.2081, 0.2386, 0.6770857143],
        [0.1959047619, 0.2644571429, 0.7279],
        [0.1707285714, 0.2919380952, 0.779247619],
        [0.1252714286, 0.3242428571, 0.8302714286],
        [0.0591333333, 0.3598333333, 0.8683333333],
        [0.0116952381, 0.3875095238, 0.8819571429],
        [0.0059571429, 0.4086142857, 0.8828428571],
        [0.0165142857, 0.4266, 0.8786333333],
        [0.032852381, 0.4430428571, 0.8719571429],
        [0.0498142857, 0.4585714286, 0.8640571429],
        [0.0629333333, 0.4736904762, 0.8554380952],
        [0.0722666667, 0.4886666667, 0.8467],
        [0.0779428571, 0.5039857143, 0.8383714286],
        [0.079347619, 0.5200238095, 0.8311809524],
        [0.0749428571, 0.5375428571, 0.8262714286],
        [0.0640571429, 0.5569857143, 0.8239571429],
        [0.0487714286, 0.5772238095, 0.8228285714],
        [0.0343428571, 0.5965809524, 0.819852381],
        [0.0265, 0.6137, 0.8135],
        [0.0238904762, 0.6286619048, 0.8037619048],
        [0.0230904762, 0.6417857143, 0.7912666667],
        [0.0227714286, 0.6534857143, 0.7767571429],
        [0.0266619048, 0.6641952381, 0.7607190476],
        [0.0383714286, 0.6742714286, 0.743552381],
        [0.0589714286, 0.6837571429, 0.7253857143],
        [0.0843, 0.6928333333, 0.7061666667],
        [0.1132952381, 0.7015, 0.6858571429],
        [0.1452714286, 0.7097571429, 0.6646285714],
        [0.1801333333, 0.7176571429, 0.6424333333],
        [0.2178285714, 0.7250428571, 0.6192619048],
        [0.2586428571, 0.7317142857, 0.5954285714],
        [0.3021714286, 0.7376047619, 0.5711857143],
        [0.3481666667, 0.7424333333, 0.5472666667],
        [0.3952571429, 0.7459, 0.5244428571],
        [0.4420095238, 0.7480809524, 0.5033142857],
        [0.4871238095, 0.7490619048, 0.4839761905],
        [0.5300285714, 0.7491142857, 0.4661142857],
        [0.5708571429, 0.7485190476, 0.4493904762],
        [0.609852381, 0.7473142857, 0.4336857143],
        [0.6473, 0.7456, 0.4188],
        [0.6834190476, 0.7434761905, 0.4044333333],
        [0.7184095238, 0.7411333333, 0.3904761905],
        [0.7524857143, 0.7384, 0.3768142857],
        [0.7858428571, 0.7355666667, 0.3632714286],
        [0.8185047619, 0.7327333333, 0.3497904762],
        [0.8506571429, 0.7299, 0.3360285714],
        [0.8824333333, 0.7274333333, 0.3217],
        [0.9139333333, 0.7257857143, 0.3062761905],
        [0.9449571429, 0.7261142857, 0.2886428571],
        [0.9738952381, 0.7313952381, 0.266647619],
        [0.9937714286, 0.7454571429, 0.240347619],
        [0.9990428571, 0.7653142857, 0.2164142857],
        [0.9955333333, 0.7860571429, 0.196652381],
        [0.988, 0.8066, 0.1793666667],
        [0.9788571429, 0.8271428571, 0.1633142857],
        [0.9697, 0.8481380952, 0.147452381],
        [0.9625857143, 0.8705142857, 0.1309],
        [0.9588714286, 0.8949, 0.1132428571],
        [0.9598238095, 0.9218333333, 0.0948380952],
        [0.9661, 0.9514428571, 0.0755333333],
        [0.9763, 0.9831, 0.0538],
    ]

    return LinearSegmentedColormap.from_list("parula", cm_data)
