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
# @Last Modified time: 2016-02-23 19:39:38
"""
Computation of the quality assessment measures on functional MRI
----------------------------------------------------------------


"""
import numpy as np


def ghost_direction(epi_data, mask_data, direction="y", ref_file=None,
                    out_file=None):
    """
    Computes the :abr:`GSR (ghost to signal ratio)` [Giannelli2010]_

    .. warning ::

      This should be used with EPI images for which the phase
      encoding direction is known.

    Parameters
    ----------
    epi_file: str
        path to epi file
    mask_file: str
        path to brain mask
    direction: str
        the direction of phase encoding (x, y, z)

    Returns
    -------
    gsr: float
        ghost to signal ratio


    .. [Giannelli2010] Giannelli et al. *Characterization of Nyquist ghost in
      EPI-fMRI acquisition sequences implemented on two clinical 1.5 T MR scanner
      systems: effect of readout bandwidth and echo spacing*. J App Clin Med Phy,
      11(4). 2010.
      doi:`10.1120/jacmp.v11i4.3237 <http://dx.doi.org/10.1120/jacmp.v11i4.3237>`.


    """
    # first we need to make a nyquist ghost mask, we do this by circle
    # shifting the original mask by N/2 and then removing the intersection
    # with the original mask
    n2_mask_data = np.zeros_like(mask_data)

    # rotate by n/2
    if direction == "x":
        n2lim = np.floor(mask_data.shape[0]/2)
        n2_mask_data[:n2lim, :, :] = mask_data[n2lim:(n2lim*2), :, :]
        n2_mask_data[n2lim:(n2lim*2), :, :] = mask_data[:n2lim, :, :]
    elif direction == "y":
        n2lim = np.floor(mask_data.shape[1]/2)
        n2_mask_data[:, :n2lim, :] = mask_data[:, n2lim:(n2lim*2), :]
        n2_mask_data[:, n2lim:(n2lim*2), :] = mask_data[:, :n2lim, :]
    elif direction == "z":
        n2lim = np.floor(mask_data.shape[2]/2)
        n2_mask_data[:, :, :n2lim] = mask_data[:, :, n2lim:(n2lim*2)]
        n2_mask_data[:, :, n2lim:(n2lim*2)] = mask_data[:, :, :n2lim]
    else:
        raise Exception("Unknown direction %s, should be x, y, or z"
                        % direction)

    # now remove the intersection with the original mask
    n2_mask_data = n2_mask_data * (1-mask_data)

    # now create a non-ghost background region, that contains 2s
    n2_mask_data = n2_mask_data + 2*(1-n2_mask_data-mask_data)

    # Save mask
    if ref_file is not None and out_file is not None:
        import nibabel as nib
        ref = nib.load(ref_file)
        out = nib.Nifti1Image(n2_mask_data, ref.get_affine(), ref.get_header())
        out.to_filename(out_file)

    # now we calculate the Ghost to signal ratio, but here we define signal
    # as the entire foreground image
    gsr = (epi_data[n2_mask_data == 1].mean(
        ) - epi_data[n2_mask_data == 2].mean())/epi_data[n2_mask_data == 0].mean()

    return gsr


def ghost_all(epi_data, mask_data):
    """Wrap the gsr computation to both possible encoding directions"""

    directions = ["x", "y"]
    gsrs = [ghost_direction(epi_data, mask_data, d) for d in directions]

    return tuple(gsrs + [None])
