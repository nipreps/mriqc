#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# @Author: oesteban
# @Date:   2016-01-05 11:29:40
# @Email:  code@oscaresteban.es
# @Last modified by:   oesteban
# @Last Modified time: 2016-04-13 08:10:35
""" Nipype interfaces to support anatomical workflow """
import os.path as op
import numpy as np
import nibabel as nb
import scipy.ndimage as nd

from nipype.interfaces.base import TraitedSpec, BaseInterface, BaseInterfaceInputSpec, File


class ArtifactMaskInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True, desc='File to be plotted')
    air_msk = File(exists=True, mandatory=True, desc='air mask')


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

    def _run_interface(self, runtime):
        imnii = nb.load(self.inputs.in_file)
        imdata = np.nan_to_num(imnii.get_data())
        # Cast to float32
        imdata = imdata.astype(np.float32)
        # Remove negative values
        imdata[imdata < 0] = 0

        airdata = nb.load(self.inputs.air_msk).get_data()
        # Run the artifact detection
        qi1_img = artifact_mask(imdata, airdata)

        fname, ext = op.splitext(op.basename(self.inputs.in_file))
        if ext == '.gz':
            fname, ext2 = op.splitext(fname)
            ext = ext2 + ext

        self._results['out_art_msk'] = op.abspath('%s_artifacts%s' % (fname, ext))
        self._results['out_air_msk'] = op.abspath('%s_noart-air%s' % (fname, ext))

        hdr = imnii.get_header().copy()
        hdr.set_data_dtype(np.uint8)
        nb.Nifti1Image(qi1_img, imnii.get_affine(), hdr).to_filename(
            self._results['out_art_msk'])

        airdata[qi1_img > 0] = 0
        nb.Nifti1Image(airdata, imnii.get_affine(), hdr).to_filename(
            self._results['out_air_msk'])
        return runtime

    def _list_outputs(self):
        return self._results

def artifact_mask(imdata, airdata):
    """Computes a mask of artifacts found in the air region"""

    if not np.issubdtype(airdata.dtype, np.integer):
        airdata[airdata < .95] = 0
        airdata[airdata > 0.] = 1

    bg_img = imdata * airdata
    # Find the background threshold (the most frequently occurring value
    # excluding 0)
    hist, bin_edges = np.histogram(bg_img[bg_img > 0], bins=128)
    bg_threshold = np.mean(bin_edges[np.argmax(hist)])


    # Apply this threshold to the background voxels to identify voxels
    # contributing artifacts.
    qi1_img = np.zeros_like(bg_img)
    qi1_img[bg_img > bg_threshold] = bg_img[bg_img > bg_threshold]

    # Create a structural element to be used in an opening operation.
    struc = nd.generate_binary_structure(3, 2)

    # Perform an a grayscale erosion operation.
    qi1_img = nd.grey_erosion(qi1_img, structure=struc).astype(np.float32)
    # Binarize and binary dilation
    qi1_img[qi1_img > 0.] = 1
    qi1_img[qi1_img < 1.] = 0
    qi1_img = nd.binary_dilation(qi1_img, structure=struc).astype(np.uint8)
    return qi1_img
