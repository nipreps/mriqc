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
# @Last Modified time: 2016-04-13 07:58:42
"""
Computation of the quality assessment measures on structural MRI



"""
from __future__ import absolute_import, division, print_function, unicode_literals
from math import pi
from six import string_types
import numpy as np
import scipy.ndimage as nd
from scipy.stats import chi

FSL_FAST_LABELS = {'csf': 1, 'gm': 2, 'wm': 3, 'bg': 0}

def snr(img, smask, nmask=None, erode=True, fglabel=1, bglabel=0):
    r"""
    Calculate the :abbr:`SNR (Signal-to-Noise Ratio)`.
    The estimation may be provided with only one foreground region in
    which the noise is computed as follows:

    .. math::

        \text{SNR} = \frac{\mu_F}{\sigma_F},

    where :math:`\mu_F` is the mean intensity of the foreground and
    :math:`\sigma_F` is the standard deviation of the same region.
    Alternatively, a background mask containing only noise can be provided.
    This must be an air mask around the head, and it should not contain artifacts.
    The computation is done following the eq. A.12 of [Dietrich2007]_, which
    includes a correction factor in the estimation of the standard deviation of
    air and its Rayleigh distribution:

    .. math::

        \text{SNR} = \frac{\mu_F}{\sqrt{\frac{2}{4-\pi}}\,\sigma_\text{air}}.


    :param numpy.ndarray img: input data
    :param numpy.ndarray fgmask: input foreground mask or segmentation
    :param numpy.ndarray bgmask: input background mask or segmentation
    :param bool erode: erode masks before computations.
    :param str fglabel: foreground label in the segmentation data.
    :param str bglabel: background label in the segmentation data.

    :return: the computed SNR for the foreground segmentation

    """
    fgmask = smask.copy()
    bgmask = None if nmask is None else nmask.copy()

    if np.issubdtype(fgmask.dtype, np.integer):
        if isinstance(fglabel, string_types):
            fglabel = FSL_FAST_LABELS[fglabel]
        if isinstance(bglabel, string_types):
            bglabel = FSL_FAST_LABELS[bglabel]

        fgmask[fgmask != fglabel] = 0
        fgmask[fgmask == fglabel] = 1

        if bgmask is not None:
            bgmask[bgmask != bglabel] = 0
            bgmask[bgmask == bglabel] = 1
    else:
        fgmask[fgmask > .95] = 1.
        fgmask[fgmask < 1.] = 0
        if bgmask is not None:
            bgmask[bgmask > .95] = 1.
            bgmask[bgmask < 1.] = 0

    if erode:
        # Create a structural element to be used in an opening operation.
        struc = nd.generate_binary_structure(3, 2)
        # Perform an opening operation on the background data.
        fgmask = nd.binary_opening(fgmask, structure=struc).astype(np.uint8)

        if bgmask is not None:
            bgmask = nd.binary_opening(bgmask, structure=struc).astype(np.uint8)

    fg_mean = np.median(img[fgmask > 0])

    if bgmask is None:
        bgmask = fgmask
        bg_mean = fg_mean
        # Manually compute sigma, using Bessel's correction (the - 1 in the normalizer)
        bg_std = np.sqrt(np.sum((img[bgmask > 0] - bg_mean) ** 2) / (np.sum(bgmask) - 1))
    else:
        bg_std = np.sqrt(2.0/(4.0 - pi)) * img[bgmask > 0].std(ddof=1)

    return fg_mean / bg_std


def cnr(img, seg, lbl=None):
    r"""
    Calculate the :abbr:`CNR (Contrast-to-Noise Ratio)` [Magnota2006]_.
    Higher values are better.

    .. math::

        \text{CNR} = \frac{|\mu_\text{GM} - \mu_\text{WM} |}{\sigma_B},

    where :math:`\sigma_B` is the standard deviation of the noise distribution within
    the air (background) mask.


    :param numpy.ndarray img: input data
    :param numpy.ndarray seg: input segmentation
    :return: the computed CNR

    """
    if lbl is None:
        lbl = FSL_FAST_LABELS

    return np.abs(img[seg == lbl['gm']].mean() - img[seg == lbl['wm']].mean()) / \
                  img[seg == lbl['bg']].std()


def cjv(img, wmmask, gmmask):
    r"""
    Calculate the :abbr:`CJV (coefficient of joint variation)`, a measure
    related to :abbr:`SNR (Signal-to-Noise Ratio)` and
    :abbr:`CNR (Contrast-to-Noise Ratio)` that is presented as a proxy for
    the :abbr:`INU (intensity non-uniformity)` artifact [Ganzetti2016]_.
    Lower is better.

    .. math::

        \text{CJV} = \frac{\sigma_\text{WM} + \sigma_\text{GM}}{\mu_\text{WM} - \mu_\text{GM}}.

    :param numpy.ndarray img: the input data
    :param numpy.ndarray wmmask: the white matter mask
    :param numpy.ndarray gmmask: the gray matter mask
    :return: the computed CJV


    """

    mu_wm = img[wmmask > .5].mean()
    mu_gm = img[gmmask > .5].mean()
    sigma_wm = img[wmmask > .5].std(ddof=1)
    sigma_gm = img[gmmask > .5].std(ddof=1)
    return (sigma_wm + sigma_gm) / (mu_wm - mu_gm)


def fber(img, seg, air=None):
    r"""
    Calculate the :abbr:`FBER (Foreground-Background Energy Ratio)`,
    defined as the mean energy of image values within the head relative
    to outside the head. Higher values are better.

    .. math::

        \text{FBER} = \frac{E[|F|^2]}{E[|B|^2]}


    :param numpy.ndarray img: input data
    :param numpy.ndarray seg: input segmentation

    """
    if air is None:
        air = np.zeros_like(seg)
        air[seg == 0] = 1

    fg_mu = (np.abs(img[seg > 0]) ** 2).mean()
    bg_mu = (np.abs(img[air > 0]) ** 2).mean()
    return fg_mu / bg_mu



def efc(img):
    """
    Calculate the :abbr:`EFC (Entropy Focus Criterion)` [Atkinson1997]_.
    Uses the Shannon entropy of voxel intensities as an indication of ghosting
    and blurring induced by head motion. Lower values are better.

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


def artifacts(img, airmask, artmask, ncoils=1, compute_qi2=False):
    """
    Detect artifacts in the image using the method described in [Mortamet2009]_.
    The **q1** is the proportion of voxels with intensity corrupted by artifacts
    normalized by the number of voxels in the background. Lower values are better.

    Optionally, it also calculates **qi2**, the distance between the distribution
    of noise voxel (non-artifact background voxels) intensities, and a
    Rician distribution.

    :param numpy.ndarray img: input data
    :param numpy.ndarray seg: input segmentation

    """

    if not np.issubdtype(airmask.dtype, np.integer):
        airmask[airmask < .95] = 0
        airmask[airmask > 0.] = 1

    if not np.issubdtype(artmask.dtype, np.integer):
        artmask[artmask < .95] = 0
        artmask[artmask > 0.] = 1

    # Count the number of voxels that remain after the opening operation.
    # These are artifacts.
    artifact_qi1 = artmask.sum() / float(airmask.sum() + artmask.sum())
    artifact_qi2 = None
    if compute_qi2:
        # Artifact-free air region
        data = img[airmask > 0]

        # Estimate data pdf
        hist, bin_edges = np.histogram(data, bins=128)
        bin_centers = [np.mean(bin_edges[i:i+1]) for i in range(len(bin_edges)-1)]

        # Find t2 (intensity at half width, left side)
        hist_max = hist[np.argmax(hist)]
        t2idx = -1
        for i in range(len(bin_centers)):
            ihw = 0.45 * hist_max
            if hist[i] > ihw:
                t2idx = i
                break

        # Fit central chi distribution
        param = chi.fit(data[data > 0], 2*ncoils, loc=bin_centers[np.argmax(hist)])
        pdf_fitted = chi.pdf(bin_centers, *param[:-2], loc=param[-2], scale=param[-1])

        # import matplotlib.pyplot as plt
        # plt.plot(bin_centers[t2idx:], hist[t2idx:])
        # plt.plot(bin_centers[t2idx:], pdf_fitted[t2idx:])
        # plt.savefig('fig.png')

        # Compute goodness-of-fit (gof)
        gof = np.abs(hist[t2idx:] - pdf_fitted[t2idx:]).sum() / airmask.sum()
        artifact_qi2 = artifact_qi1 + gof

    return artifact_qi1, artifact_qi2

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
