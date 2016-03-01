#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
# pylint: disable=no-member
#
# @Author: oesteban
# @Date:   2016-01-05 11:29:40
# @Email:  code@oscaresteban.es
# @Last modified by:   oesteban
# @Last Modified time: 2016-02-29 11:40:36
"""
Computation of the quality assessment measures on structural MRI



"""

import numpy as np
from six import string_types
import scipy.ndimage as nd

FSL_FAST_LABELS = {'csf': 1, 'gm': 2, 'wm': 3, 'bg': 0}

def snr(img, seg, fglabel, bglabel='bg'):
    r"""
    Calculate the :abbr:`SNR (Signal-to-Noise Ratio)`

    .. math::

        \text{SNR} = \frac{\mu_F}{\sigma_B}


    where :math:`\mu_F` is the mean intensity of the foreground and
    :math:`\sigma_B` is the standard deviation of the background,
    where the noise is computed.

    :param numpy.ndarray img: input data
    :param numpy.ndarray seg: input segmentation
    :param str fglabel: foreground label in the segmentation data.
    :param str bglabel: background label in the segmentation data.

    :return: the computed SNR for the foreground segmentation

    """
    if isinstance(fglabel, string_types):
        fglabel = FSL_FAST_LABELS[fglabel]
    if isinstance(bglabel, string_types):
        bglabel = FSL_FAST_LABELS[bglabel]

    fg_mean = img[seg == fglabel].mean()
    bg_std = img[seg == bglabel].std()
    return fg_mean / bg_std


def cnr(img, seg, lbl=None):
    r"""
    Calculate the :abbr:`CNR (Contrast-to-Noise Ratio)`

    .. math::

        \text{CNR} = \frac{|\mu_\text{GM} - \mu_\text{WM} |}{\sigma_B}


    :param numpy.ndarray img: input data
    :param numpy.ndarray seg: input segmentation
    :return: the computed CNR

    """
    if lbl is None:
        lbl = FSL_FAST_LABELS

    return np.abs(img[seg == lbl['gm']].mean() - img[seg == lbl['wm']].mean()) / \
                  img[seg == lbl['bg']].std()



def fber(img, seg, fglabel=None, bglabel=0):
    r"""
    Calculate the :abbr:`FBER (Foreground-Background Energy Ratio)`

    .. math::

        \text{FBER} = \frac{E[|F|^2]}{E[|B|^2]}


    :param numpy.ndarray img: input data
    :param numpy.ndarray seg: input segmentation

    """
    if fglabel is not None:
        fgdata = img[seg == FSL_FAST_LABELS[fglabel]]
    else:
        fgdata = img[seg != bglabel]

    fg_mu = (np.abs(fgdata) ** 2).mean()
    bg_mu = (np.abs(img[seg == bglabel]) ** 2).mean()
    return fg_mu / bg_mu



def efc(img):
    """
    Calculate the :abbr:`EFC (Entropy Focus Criterion)` [Atkinson1997]_

    The original equation is normalized by the maximum entropy, so that the
    :abbr:`EFC (Entropy Focus Criterion)` can be compared across images with
    different dimensions.

    :param numpy.ndarray img: input data

    """

    # Calculate the maximum value of the EFC (which occurs any time all
    # voxels have the same value)
    efc_max = 1.0 * np.prod(img.shape) * (1.0 / np.sqrt(np.prod(img.shape))) * \
                np.log(1.0 / np.sqrt(np.prod(img.shape)))

    # Calculate the total image energy
    b_max = np.sqrt((img**2).sum())

    # Calculate EFC (add 1e-16 to the image data to keep log happy)
    return (1.0 / efc_max) * np.sum((img / b_max) * np.log((img + 1e-16) / b_max))


def artifacts(img, seg, calculate_qi2=False, bglabel=0):
    """
    Detect artifacts in the image using the method described in [Mortamet2009]_.
    Calculates QI1, the fraction of total voxels that within artifacts.

    Optionally, it also calculates QI2, the distance between the distribution
    of noise voxel (non-artifact background voxels) intensities, and a
    Rician distribution.

    :param numpy.ndarray img: input data
    :param numpy.ndarray seg: input segmentation

    """
    bg_mask = np.zeros_like(img, dtype=np.uint8)
    bg_mask[seg == bglabel] = 1
    bg_img = img * bg_mask

    # Find the background threshold (the most frequently occurring value
    # excluding 0)
    hist, bin_edges = np.histogram(bg_img[bg_img > 0], bins=256)
    bg_threshold = np.mean(bin_edges[np.argmax(hist)])

    # Apply this threshold to the background voxels to identify voxels
    # contributing artifacts.
    bg_img[bg_img <= bg_threshold] = 0
    bg_img[bg_img != 0] = 1

    # Create a structural element to be used in an opening operation.
    struct_elmnt = np.zeros((3, 3, 3))
    struct_elmnt[0, 1, 1] = 1
    struct_elmnt[1, 1, :] = 1
    struct_elmnt[1, :, 1] = 1
    struct_elmnt[2, 1, 1] = 1

    # Perform an opening operation on the background data.
    bg_img = nd.binary_opening(bg_img, structure=struct_elmnt).astype(np.uint8)

    # Count the number of voxels that remain after the opening operation.
    # These are artifacts.
    artifact_qi1 = bg_img.sum() / float(bg_mask.sum())

    if calculate_qi2:
        raise NotImplementedError

    return artifact_qi1, None

def volume_fraction(pvms):
    """
    Computes the :abbr:`ICV (intracranial volume)` fractions
    corresponding to the (partial volume maps).

    :param list pvms: list of :code:`numpy.ndarray` of partial volume maps.

    """
    tissue_vfs = {}
    total = 0
    for k, lid in list(FSL_FAST_LABELS.items()):
        if lid == 0:
            continue
        tissue_vfs[k] = pvms[lid - 1].sum()
        total += tissue_vfs[k]

    for k in tissue_vfs.keys():
        tissue_vfs[k] /= total

    return tissue_vfs

def rpve(pvms, seg):
    """
    Computes the :abbr:`rPVe (residual partial voluming error)`
    of each tissue class.
    """
    pvfs = {}
    for k, lid in list(FSL_FAST_LABELS.items()):
        if lid == 0:
            continue
        pvmap = pvms[lid - 1][seg == lid]
        pvmap[pvmap < 0.] = 0.
        pvmap[pvmap >= 1.] = 0.
        upth = np.percentile(pvmap[pvmap > 0], 98)
        loth = np.percentile(pvmap[pvmap > 0], 2)
        pvmap[pvmap < loth] = 0
        pvmap[pvmap > upth] = 0
        pvfs[k] = pvmap[pvmap > 0].sum()
    return pvfs

def summary_stats(img, pvms):
    r"""
    Estimates the mean, the standard deviation, the 95\%
    and the 5\% percentiles of each tissue distribution.
    """
    mean = {}
    stdv = {}
    p95 = {}
    p05 = {}

    if np.array(pvms).ndim == 4:
        pvms.insert(0, np.array(pvms).sum(axis=0))
    elif np.array(pvms).ndim == 3:
        bgpvm = np.ones_like(pvms)
        pvms = [bgpvm - pvms, pvms]
    else:
        raise RuntimeError('Incorrect image dimensions (%d)' %
            np.array(pvms).ndim)

    if len(pvms) == 4:
        labels = list(FSL_FAST_LABELS.items())
    elif len(pvms) == 2:
        labels = zip(['bg', 'fg'], range(2))

    for k, lid in labels:
        im_lid = pvms[lid] * img
        mean[k] = im_lid[im_lid > 0].mean()
        stdv[k] = im_lid[im_lid > 0].std()
        p95[k] = np.percentile(im_lid[im_lid > 0], 95)
        p05[k] = np.percentile(im_lid[im_lid > 0], 5)

    return mean, stdv, p95, p05
