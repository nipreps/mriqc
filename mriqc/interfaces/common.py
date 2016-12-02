#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

from __future__ import print_function, division, absolute_import, unicode_literals
from os import path as op
import numpy as np
import nibabel as nb

from nipype import logging
from nipype.interfaces.base import traits, TraitedSpec, BaseInterfaceInputSpec, File
from .base import MRIQCBaseInterface

IFLOGGER = logging.getLogger('interface')

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

    List of nifti datatypes:

    .. note: original Analyze 7.5 types

      DT_NONE                    0
      DT_UNKNOWN                 0     / what it says, dude           /
      DT_BINARY                  1     / binary (1 bit/voxel)         /
      DT_UNSIGNED_CHAR           2     / unsigned char (8 bits/voxel) /
      DT_SIGNED_SHORT            4     / signed short (16 bits/voxel) /
      DT_SIGNED_INT              8     / signed int (32 bits/voxel)   /
      DT_FLOAT                  16     / float (32 bits/voxel)        /
      DT_COMPLEX                32     / complex (64 bits/voxel)      /
      DT_DOUBLE                 64     / double (64 bits/voxel)       /
      DT_RGB                   128     / RGB triple (24 bits/voxel)   /
      DT_ALL                   255     / not very useful (?)          /

    .. note: added names for the same data types

      DT_UINT8                   2
      DT_INT16                   4
      DT_INT32                   8
      DT_FLOAT32                16
      DT_COMPLEX64              32
      DT_FLOAT64                64
      DT_RGB24                 128


    .. note: new codes for NIFTI

      DT_INT8                  256     / signed char (8 bits)         /
      DT_UINT16                512     / unsigned short (16 bits)     /
      DT_UINT32                768     / unsigned int (32 bits)       /
      DT_INT64                1024     / long long (64 bits)          /
      DT_UINT64               1280     / unsigned long long (64 bits) /
      DT_FLOAT128             1536     / long double (128 bits)       /
      DT_COMPLEX128           1792     / double pair (128 bits)       /
      DT_COMPLEX256           2048     / long double pair (256 bits)  /
      NIFTI_TYPE_UINT8           2 /! unsigned char. /
      NIFTI_TYPE_INT16           4 /! signed short. /
      NIFTI_TYPE_INT32           8 /! signed int. /
      NIFTI_TYPE_FLOAT32        16 /! 32 bit float. /
      NIFTI_TYPE_COMPLEX64      32 /! 64 bit complex = 2 32 bit floats. /
      NIFTI_TYPE_FLOAT64        64 /! 64 bit float = double. /
      NIFTI_TYPE_RGB24         128 /! 3 8 bit bytes. /
      NIFTI_TYPE_INT8          256 /! signed char. /
      NIFTI_TYPE_UINT16        512 /! unsigned short. /
      NIFTI_TYPE_UINT32        768 /! unsigned int. /
      NIFTI_TYPE_INT64        1024 /! signed long long. /
      NIFTI_TYPE_UINT64       1280 /! unsigned long long. /
      NIFTI_TYPE_FLOAT128     1536 /! 128 bit float = long double. /
      NIFTI_TYPE_COMPLEX128   1792 /! 128 bit complex = 2 64 bit floats. /
      NIFTI_TYPE_COMPLEX256   2048 /! 256 bit complex = 2 128 bit floats /

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

            if datatype == 1:
                IFLOGGER.warn('Input image %s has a suspicious data type "%s"',
                              self.inputs.in_file, hdr.get_data_dtype())

            # signed char and bool to uint8
            if datatype == 1 or datatype == 2 or datatype == 256:
                dtype = np.uint8

            # int16 to uint16
            elif datatype == 4:
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
