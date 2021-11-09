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
from mriqc import config, messages
from mriqc.interfaces import data_types
from nipype.interfaces.base import (
    BaseInterfaceInputSpec,
    File,
    SimpleInterface,
    TraitedSpec,
    traits,
)

#: Output file name format.
OUT_FILE = "{prefix}_conformed{ext}"

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

    in_file = File(exists=True, mandatory=True, desc="input image")
    check_ras = traits.Bool(True, usedefault=True, desc="check that orientation is RAS")
    check_dtype = traits.Bool(True, usedefault=True, desc="check data type")


class ConformImageOutputSpec(TraitedSpec):
    """
    Output specification for the :class:`ConformImage` interface.
    """

    out_file = File(exists=True, desc="output conformed file")


class ConformImage(SimpleInterface):
    f"""
    Conforms an input image.

    List of nifti datatypes:

    .. note: Original Analyze 7.5 types

       {data_types.ANALYZE_75}

    .. note: Added names for the same data types

       {data_types.ADDED}

    .. note: New codes for NIFTI

       {data_types.NEW_CODES}

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
        datatype = int(header["datatype"])
        self._warn_suspicious_dtype(datatype)
        try:
            dtype = NUMPY_DTYPE[datatype]
        except KeyError:
            return nii
        else:
            header.set_data_dtype(dtype)
            converted = nii.get_data().astype(dtype)
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
        if ext == ".gz":
            out_file, ext2 = op.splitext(out_file)
            ext = ext2 + ext
        out_file_name = OUT_FILE.format(prefix=out_file, ext=ext)
        self._results["out_file"] = op.abspath(out_file_name)
        nii.to_filename(self._results["out_file"])

        return runtime
