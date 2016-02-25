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
# @Last Modified time: 2016-02-23 16:11:11
"""
Computation of the quality assessment measures on structural MRI
----------------------------------------------------------------


"""

import numpy as np
import scipy.ndimage as nd

FSL_FAST_LABELS = {'csf': 1, 'gm': 2, 'wm': 3, 'bg': 0}

def snr(img, seg, fglabel, bglabel='bg'):
    r"""
    Calculate the :abr:`SNR (Signal-to-Noise Ratio)`

    .. math::

        \text{SNR} = \frac{\mu_F}{\sigma_B}


    where :math:`\mu_F` is the mean intensity of the foreground and
    :math>`\sigma_B` is the standard deviation of the background,
    where the noise is computed.

    """
    fg_mean = img[seg == FSL_FAST_LABELS[fglabel]].mean()
    bg_std = img[seg == FSL_FAST_LABELS[bglabel]].std()
    return fg_mean / bg_std


def cnr(img, seg, lbl=None):
    r"""
    Calculate the :abr:`CNR (Contrast-to-Noise Ratio)`

    .. math::

        \text{CNR} = \frac{|\mu_\text{GM} - \mu_\text{WM} |}{\sigma_B}

    """
    if lbl is None:
        lbl = FSL_FAST_LABELS

    return np.abs(img[seg == lbl['gm']].mean() - img[seg == lbl['wm']].mean()) / \
                  img[seg == lbl['bg']].std()



def fber(img, seg, fglabel=None, bglabel=0):
    r"""
    Calculate the :abr:`FBER (Foreground-Background Energy Ratio)`

    .. math::

        \text{FBER} = \frac{E[|F|^2]}{E[|B|^2]}


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
    Calculate the :abr:`EFC (Entropy Focus Criterion)` [Atkinson1997]_

    The original equation is normalized by the maximum entropy, so that the
    :abr:`EFC (Entropy Focus Criterion)` can be compared across images with
    different dimensions.

    .. [Atkinson1997] Atkinson, D.; Hill, D.L.G.; Stoyle, P.N.R.; Summers, P.E.; Keevil, S.F.,
      *Automatic correction of motion artifacts in magnetic resonance images using an entropy
      focus criterion*, IEEE Trans Med Imag 16(6):903-910, 1997.
      doi:`10.1109/42.650886 <http://dx.doi.org/10.1109/42.650886>`


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

    .. [Mortamet2009] Mortamet B et al., *Automatic quality assessment in
      structural brain magnetic resonance imaging*, Mag Res Med 62(2):365-372,
      2009. doi:`10.1002/mrm.21992 <http://dx.doi.org/10.1002/mrm.21992>`

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
    Computes the :abr:`ICV (intracranial volume)` fractions
    corresponding to the (partial volume maps).
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
    Computes the :abr:`rPVe (residual partial voluming error)`
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

    bgpvm = np.array(pvms).sum(axis=0)
    pvms.insert(0, bgpvm)

    for k, lid in list(FSL_FAST_LABELS.items()):
        im_lid = pvms[lid] * img
        mean[k] = im_lid[im_lid > 0].mean()
        stdv[k] = im_lid[im_lid > 0].std()
        p95[k] = np.percentile(im_lid[im_lid > 0], 95)
        p05[k] = np.percentile(im_lid[im_lid > 0], 5)

    return mean, stdv, p95, p05
