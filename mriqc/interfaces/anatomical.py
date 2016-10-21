#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# @Author: oesteban
# @Date:   2016-01-05 11:29:40
# @Email:  code@oscaresteban.es
# @Last modified by:   oesteban
# @Last Modified time: 2016-10-07 14:37:52
""" Nipype interfaces to support anatomical workflow """
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from __future__ import unicode_literals
import os.path as op
import numpy as np
import nibabel as nb
import scipy.ndimage as nd

from nipype.interfaces.base import TraitedSpec, BaseInterface, BaseInterfaceInputSpec, File


class ArtifactMaskInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True, desc='File to be plotted')
    head_mask = File(exists=True, mandatory=True, desc='head mask')
    nasion_post_mask = File(exists=True, mandatory=True,
                            desc='nasion to posterior of cerebellum mask')


class ArtifactMaskOutputSpec(TraitedSpec):
    out_art_msk = File(exists=True, desc='output artifacts mask')
    out_air_msk = File(exists=True, desc='output artifacts mask, without artifacts')


class ArtifactMask(BaseInterface):
    """
    Computes the artifact mask using the method described in [Mortamet2009]_.
    """
    input_spec = ArtifactMaskInputSpec
    output_spec = ArtifactMaskOutputSpec

    def __init__(self, **inputs):
        self._results = {}
        super(ArtifactMask, self).__init__(**inputs)

    def _list_outputs(self):
        return self._results

    def _run_interface(self, runtime):
        imnii = nb.load(self.inputs.in_file)
        imdata = np.nan_to_num(imnii.get_data().astype(np.float32))

        # Remove negative values
        imdata[imdata < 0] = 0

        hmdata = nb.load(self.inputs.head_mask).get_data()
        npdata = nb.load(self.inputs.nasion_post_mask).get_data()

        # Invert head mask
        airdata = np.ones_like(hmdata, dtype=np.uint8)
        airdata[hmdata == 1] = 0

        # Calculate distance to border
        dist = nd.morphology.distance_transform_edt(airdata)

        # Apply nasion-to-posterior mask
        airdata[npdata == 1] = 0
        dist[npdata == 1] = 0
        dist /= dist.max()

        # Run the artifact detection
        qi1_img = artifact_mask(imdata, airdata, dist)

        fname, ext = op.splitext(op.basename(self.inputs.in_file))
        if ext == '.gz':
            fname, ext2 = op.splitext(fname)
            ext = ext2 + ext

        self._results['out_art_msk'] = op.abspath('{}_artifacts{}'.format(fname, ext))
        self._results['out_air_msk'] = op.abspath('{}_noart-air{}'.format(fname, ext))

        hdr = imnii.get_header().copy()
        hdr.set_data_dtype(np.uint8)
        nb.Nifti1Image(qi1_img, imnii.get_affine(), hdr).to_filename(
            self._results['out_art_msk'])

        airdata[qi1_img > 0] = 0
        nb.Nifti1Image(airdata, imnii.get_affine(), hdr).to_filename(
            self._results['out_air_msk'])
        return runtime


def artifact_mask(imdata, airdata, distance):
    """Computes a mask of artifacts found in the air region"""
    import nibabel as nb

    if not np.issubdtype(airdata.dtype, np.integer):
        airdata[airdata < .95] = 0
        airdata[airdata > 0.] = 1

    bg_img = imdata * airdata
    # Find the background threshold (the most frequently occurring value
    # excluding 0)
    # CHANGED - to the 75 percentile
    bg_threshold = np.percentile(bg_img[airdata > 0], 75)

    # Apply this threshold to the background voxels to identify voxels
    # contributing artifacts.
    qi1_img = np.zeros_like(bg_img)
    qi1_img[bg_img > bg_threshold] = 1
    qi1_img[distance < .10] = 0

    # Create a structural element to be used in an opening operation.
    struc = nd.generate_binary_structure(3, 1)
    qi1_img = nd.binary_opening(qi1_img, struc).astype(np.uint8)
    qi1_img[airdata <= 0] = 0

    return qi1_img
