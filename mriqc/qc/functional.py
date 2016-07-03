#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
# pylint: disable=no-member
#
# @Author: oesteban
# @Date:   2016-02-23 19:25:39
# @Email:  code@oscaresteban.es
# @Last Modified by:   oesteban
# @Last Modified time: 2016-02-29 11:43:16
"""
Computation of the quality assessment measures on functional MRI



"""
import os.path as op
import numpy as np
import nibabel as nb


def gsr(epi_data, mask, direction="y", ref_file=None, out_file=None):
    """
    Computes the :abbr:`GSR (ghost to signal ratio)` [Giannelli2010]_. The
    procedure is as follows:

      #. Create a Nyquist ghost mask by circle-shifting the original mask by :math:`N/2`.

      #. Rotate by :math:`N/2`

      #. Remove the intersection with the original mask

      #. Generate a non-ghost background

      #. Calculate the :abbr:`GSR (ghost to signal ratio)`


    .. warning ::

      This should be used with EPI images for which the phase
      encoding direction is known.

    :param str epi_file: path to epi file
    :param str mask_file: path to brain mask
    :param str direction: the direction of phase encoding (x, y, all)
    :return: the computed gsr

    """

    direction = direction.lower()
    if direction[-1] not in ['x', 'y', 'all']:
        raise Exception("Unknown direction %s, should be one of x, -x, y, -y, all"
                        % direction)

    if direction == 'all':
        result = []
        for newdir in ['x', 'y']:
            ofile = None
            if out_file is not None:
                fname, ext = op.splitext(ofile)
                if ext == '.gz':
                    fname, ext2 = op.splitext(fname)
                    ext = ext2 + ext
                ofile = '%s_%s%s' % (fname, newdir, ext)
            result += [gsr(epi_data, mask, newdir,
                           ref_file=ref_file, out_file=ofile)]
        return result

    # Step 1
    n2_mask = np.zeros_like(mask)

    # Step 2
    if direction == "x":
        n2max = mask.shape[0]
        n2lim = int(np.floor(n2max/2))
        n2_mask[:n2lim, :, :] = mask[n2lim:n2max, :, :]
        n2_mask[n2lim:n2max, :, :] = mask[:n2lim, :, :]
    elif direction == "y":
        n2max = mask.shape[1]
        n2lim = int(np.floor(n2max/2))
        n2_mask[:, :n2lim, :] = mask[:, n2lim:n2max, :]
        n2_mask[:, n2lim:n2max, :] = mask[:, :n2lim, :]
    elif direction == "z":
        n2max = mask.shape[2]
        n2lim = int(np.floor(n2max/2))
        n2_mask[:, :, :n2lim] = mask[:, :, n2lim:n2max]
        n2_mask[:, :, n2lim:n2max] = mask[:, :, :n2lim]

    # Step 3
    n2_mask = n2_mask * (1-mask)

    # Step 4: non-ghost background region is labeled as 2
    n2_mask = n2_mask + 2 * (1 - n2_mask - mask)

    # Save mask
    if ref_file is not None and out_file is not None:
        ref = nb.load(ref_file)
        out = nb.Nifti1Image(n2_mask, ref.get_affine(), ref.get_header())
        out.to_filename(out_file)

    # Step 5: signal is the entire foreground image
    ghost = epi_data[n2_mask == 1].mean() - epi_data[n2_mask == 2].mean()
    signal = epi_data[n2_mask == 0].mean()
    return float(ghost/signal)


def dvars(in_file, in_mask, output_all=False, out_file=None):
    """
    Compute the mean :abbr:`DVARS (D referring to temporal
    derivative of timecourses, VARS referring to RMS variance over voxels)`
    [Power2012]_.

    Particularly, the *standardized* :abbr:`DVARS (D referring to temporal
    derivative of timecourses, VARS referring to RMS variance over voxels)`
    [Nichols2013]_ are computed.

    .. note:: Implementation details

      Uses the implementation of the `Yule-Walker equations
      from nitime
      <http://nipy.org/nitime/api/generated/nitime.algorithms.autoregressive.html\
#nitime.algorithms.autoregressive.AR_est_YW>`_
      for the :abbr:`AR (auto-regressive)` filtering of the fMRI signal.

    :param numpy.ndarray func: functional data, after head-motion-correction.
    :param numpy.ndarray mask: a 3D mask of the brain
    :param bool output_all: write out all dvars
    :param str out_file: a path to which the standardized dvars should be saved.
    :return: the standardized DVARS

    """
    import os.path as op
    import numpy as np
    import nibabel as nb
    from nitime.algorithms import AR_est_YW
    from mriqc.qc.functional import zero_variance

    func = nb.load(in_file).get_data()
    mask = nb.load(in_mask).get_data()

    if len(func.shape) != 4:
        raise RuntimeError(
            "Input fMRI dataset should be 4-dimensional" % func)

    # Remove zero-variance voxels across time axis
    zv_mask = zero_variance(func, mask)
    idx = np.where(zv_mask > 0)
    mfunc = func[idx[0], idx[1], idx[2], :]

    # Robust standard deviation
    func_sd = (np.percentile(mfunc, 75) -
               np.percentile(mfunc, 25)) / 1.349

    # Demean
    mfunc -= mfunc.mean(axis=1)[..., np.newaxis]

    # AR1
    ak_coeffs = np.apply_along_axis(AR_est_YW, 1, mfunc, 1)

    # Predicted standard deviation of temporal derivative
    func_sd_pd = np.squeeze(np.sqrt((2 * (1 - ak_coeffs[:, 0])).tolist()) * func_sd)
    diff_sd_mean = func_sd_pd[func_sd_pd > 0].mean()

    # Compute temporal difference time series
    func_diff = np.diff(mfunc, axis=1)

    # DVARS (no standardization)
    dvars_nstd = func_diff.std(axis=0)

    # standardization
    dvars_stdz = dvars_nstd / diff_sd_mean

    # voxelwise standardization
    diff_vx_stdz = func_diff / np.array([func_sd_pd] * func_diff.shape[-1]).T
    dvars_vx_stdz = diff_vx_stdz.std(1, ddof=1)

    if output_all:
        gendvars = np.vstack((dvars_stdz, dvars_nstd, dvars_vx_stdz))
    else:
        gendvars = dvars_stdz.reshape(len(dvars_stdz), 1)

    if out_file is None:
        fname, ext = op.splitext(op.basename(in_file))
        if ext == '.gz':
            fname, _ = op.splitext(fname)
        fname += '_dvars.txt'
        out_file = op.abspath(fname)

    np.savetxt(out_file, gendvars, fmt='%.12f')
    return out_file


def summary_fd(fd_movpar, fd_thres=1.0):
    """
    Generates a dictionary with the mean FD, the number of FD timepoints above
    fd_thres, and the percentage of FD timepoints above the fd_thres
    """
    fddata = np.loadtxt(fd_movpar)
    num_fd = np.float((fddata > fd_thres).sum())
    out_dict = {
        'mean_fd': float(fddata.mean()),
        'num_fd': int(num_fd),
        'perc_fd': float(num_fd * 100 / (len(fddata) + 1))
    }
    return out_dict

def gcor(func, mask):
    """
    Compute the :abbr:`GCOR (global correlation)`.

    :param numpy.ndarray func: input fMRI dataset, after motion correction
    :param numpy.ndarray mask: 3D brain mask
    :return: the computed GCOR value

    """
    from scipy.stats.mstats import zscore
    # Remove zero-variance voxels across time axis
    tv_mask = zero_variance(func, mask)
    idx = np.where(tv_mask > 0)
    zscores = zscore(func[idx[0], idx[1], idx[2], :], axis=1)
    avg_ts = zscores.mean(axis=0)
    return float(avg_ts.transpose().dot(avg_ts) / len(avg_ts))

def zero_variance(func, mask):
    """
    Mask out voxels with zero variance across t-axis

    :param numpy.ndarray func: input fMRI dataset, after motion correction
    :param numpy.ndarray mask: 3D brain mask
    :return: the 3D mask of voxels with nonzero variance across :math:`t`.
    :rtype: numpy.ndarray

    """
    idx = np.where(mask > 0)
    func = func[idx[0], idx[1], idx[2], :]
    tvariance = func.var(axis=1)
    tv_mask = np.zeros_like(tvariance)
    tv_mask[tvariance > 0] = 1

    newmask = np.zeros_like(mask)
    newmask[idx] = tv_mask
    return newmask
