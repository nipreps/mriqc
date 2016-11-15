#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

from __future__ import print_function, division, absolute_import, unicode_literals
from os import path as op
import numpy as np
import nibabel as nb

from .base import MRIQCBaseInterface
from nipype.interfaces.base import traits, TraitedSpec, BaseInterfaceInputSpec, File



class ConformImageInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True, desc='input image')
    check_ras = traits.Bool(True, usedefault=True,
                            desc='check that orientation is RAS')
    check_dtype = traits.Bool(True, usedefault=True,
                              desc='check data type')


class ConformImageOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='output conformed file')


class ConformImage(MRIQCBaseInterface):

    """
    Conforms an input image

    """
    input_spec = ConformImageInputSpec
    output_spec = ConformImageOutputSpec

    def _run_interface(self, runtime):
        # load image
        nii = nb.load(self.inputs.in_file)
        hdr = nii.get_header().copy()

        if self.inputs.check_ras:
            nii = nb.as_closest_canonical(nii)

        if self.inputs.check_dtype:
            changed = True
            datatype = int(hdr['datatype'])

            # signed char and bool to uint8
            if datatype == 4 or datatype == 2:
                dtype = np.uint8

            # int to uint16
            elif datatype == 256:
                dtype = np.uint16

            # Signed long, long long, etc to uint32
            elif datatype == 8 or datatype == 1024 or datatype == 1280:
                dtype = np.uint32

            # Floats over 32 bits
            elif datatype == 64 or datatype == 1536:
                dtype = np.float32
            else:
                changed = False

            if changed:
                hdr.set_data_dtype(dtype)
                nii = nb.Nifti1Image(nii.get_data().astype(dtype),
                                     nii.get_affine(), hdr)

        # Generate name
        out_file, ext = op.splitext(op.basename(self.inputs.in_file))
        if ext == '.gz':
            out_file, ext2 = op.splitext(out_file)
            ext = ext2 + ext

        self._results['out_file'] = op.abspath('{}_conformed{}'.format(out_file, ext))
        nii.to_filename(self._results['out_file'])
        return runtime
