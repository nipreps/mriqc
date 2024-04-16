# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# Copyright 2021 The NiPreps Developers <nipreps@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# We support and encourage derived works from this project, please read
# about our expectations at
#
#     https://www.nipreps.org/community/licensing/
#
"""
Definition of the :class:`ConformImage` interface.
"""

from os import path as op

import nibabel as nib
import numpy as np
from nipype.interfaces.base import (
    BaseInterfaceInputSpec,
    File,
    SimpleInterface,
    TraitedSpec,
    traits,
)

from mriqc import config, messages

#: Output file name format.
OUT_FILE = '{prefix}_conformed{ext}'

#: NIfTI header datatype code to numpy dtype.
NUMPY_DTYPE = {
    1: np.uint8,
    2: np.uint8,
    4: np.uint16,
    8: np.uint32,
    64: np.float32,
    256: np.uint8,
    1024: np.uint32,
    1280: np.uint32,
    1536: np.float32,
}


class ConformImageInputSpec(BaseInterfaceInputSpec):
    """
    Input specification for the :class:`ConformImage` interface.
    """

    in_file = File(exists=True, mandatory=True, desc='input image')
    check_ras = traits.Bool(True, usedefault=True, desc='check that orientation is RAS')
    check_dtype = traits.Bool(True, usedefault=True, desc='check data type')


class ConformImageOutputSpec(TraitedSpec):
    """
    Output specification for the :class:`ConformImage` interface.
    """

    out_file = File(exists=True, desc='output conformed file')


class ConformImage(SimpleInterface):
    """
    Conforms an input image.

    List of nifti datatypes:

    .. note: Original Analyze 7.5 types

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

    .. note: Added names for the same data types

          DT_UINT8                   2
          DT_INT16                   4
          DT_INT32                   8
          DT_FLOAT32                16
          DT_COMPLEX64              32
          DT_FLOAT64                64
          DT_RGB24                 128

    .. note: New codes for NIfTI

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

    def _warn_suspicious_dtype(self, dtype: int) -> None:
        """
        Warns about binary type *nii* images.

        Parameters
        ----------
        dtype : int
            NIfTI header datatype
        """
        if dtype == 1:
            dtype_message = messages.SUSPICIOUS_DATA_TYPE.format(
                in_file=self.inputs.in_file, dtype=dtype
            )
            config.loggers.interface.warning(dtype_message)

    def _check_dtype(self, nii: nib.Nifti1Image) -> nib.Nifti1Image:
        """
        Checks the NIfTI header datatype and converts the data to the matching
        numpy dtype.

        Parameters
        ----------
        nii : nib.Nifti1Image
            Input image

        Returns
        -------
        nib.Nifti1Image
            Converted input image
        """
        header = nii.header.copy()
        datatype = int(header['datatype'])
        self._warn_suspicious_dtype(datatype)
        try:
            dtype = NUMPY_DTYPE[datatype]
        except KeyError:
            return nii
        else:
            header.set_data_dtype(dtype)
            converted = np.asanyarray(nii.dataobj, dtype=dtype)
            return nib.Nifti1Image(converted, nii.affine, header)

    def _run_interface(self, runtime):
        """
        Execute this interface with the provided runtime.

        TODO: Is the *runtime* argument required? It doesn't seem to be used
              anywhere.

        Parameters
        ----------
        runtime : Any
            Execution runtime ?

        Returns
        -------
        Any
            Execution runtime ?
        """
        # Squeeze 4th dimension if possible (#660)
        nii = nib.squeeze_image(nib.load(self.inputs.in_file))

        if self.inputs.check_ras:
            nii = nib.as_closest_canonical(nii)

        if self.inputs.check_dtype:
            nii = self._check_dtype(nii)

        # Generate name
        out_file, ext = op.splitext(op.basename(self.inputs.in_file))
        if ext == '.gz':
            out_file, ext2 = op.splitext(out_file)
            ext = ext2 + ext
        out_file_name = OUT_FILE.format(prefix=out_file, ext=ext)
        self._results['out_file'] = op.abspath(out_file_name)
        nii.to_filename(self._results['out_file'])

        return runtime
