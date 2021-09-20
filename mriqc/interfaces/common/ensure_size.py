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
Definition of the :class:`EnsureSize` interface.
"""
from os import path as op

import nibabel as nib
import numpy as np
from mriqc import config, messages
from nipype.interfaces.ants import ApplyTransforms
from nipype.interfaces.base import (
    BaseInterfaceInputSpec,
    File,
    SimpleInterface,
    TraitedSpec,
    isdefined,
    traits,
)
from pkg_resources import resource_filename as pkgrf

OUT_FILE_NAME = "{prefix}_resampled{ext}"
OUT_MASK_NAME = "{prefix}_resmask{ext}"
REF_FILE_NAME = "resample_ref.nii.gz"
REF_MASK_NAME = "mask_ref.nii.gz"


class EnsureSizeInputSpec(BaseInterfaceInputSpec):
    """
    Input specification for the :class:`EnsureSize` interface.
    """

    in_file = File(exists=True, copyfile=False, mandatory=True, desc="input image")
    in_mask = File(exists=True, copyfile=False, desc="input mask")
    pixel_size = traits.Float(2.0, usedefault=True, desc="desired pixel size (mm)")


class EnsureSizeOutputSpec(TraitedSpec):
    """
    Output specification for the :class:`EnsureSize` interface.
    """

    out_file = File(exists=True, desc="output image")
    out_mask = File(exists=True, desc="output mask")


class EnsureSize(SimpleInterface):
    """
    Checks the size of the input image and resamples it to have `pixel_size`.
    """

    input_spec = EnsureSizeInputSpec
    output_spec = EnsureSizeOutputSpec

    def _check_size(self, nii: nib.Nifti1Image) -> bool:
        zooms = nii.header.get_zooms()
        size_diff = np.array(zooms[:3]) - (self.inputs.pixel_size - 0.1)
        if np.all(size_diff >= -1e-3):
            config.loggers.interface.info(messages.VOXEL_SIZE_OK)
            return True
        else:
            small_voxel_message = messages.VOXEL_SIZE_SMALL.format(
                *zooms[:3], self.inputs.pixel_size, *size_diff
            )
            config.loggers.interface.info(small_voxel_message)
            return False

    def _run_interface(self, runtime):
        nii = nib.load(self.inputs.in_file)
        size_ok = self._check_size(nii)
        if size_ok:
            self._results["out_file"] = self.inputs.in_file
            if isdefined(self.inputs.in_mask):
                self._results["out_mask"] = self.inputs.in_mask
        else:
            # Figure out new matrix
            # 1) Get base affine
            aff_base = nii.header.get_base_affine()
            aff_base_inv = np.linalg.inv(aff_base)

            # 2) Find center pixel in mm
            center_idx = (np.array(nii.shape[:3]) - 1) * 0.5
            center_mm = aff_base.dot(center_idx.tolist() + [1])

            # 3) Find extent of each dimension
            min_mm = aff_base.dot([-0.5, -0.5, -0.5, 1])
            max_mm = aff_base.dot((np.array(nii.shape[:3]) - 0.5).tolist() + [1])
            extent_mm = np.abs(max_mm - min_mm)[:3]

            # 4) Find new matrix size
            new_size = np.array(extent_mm / self.inputs.pixel_size, dtype=int)

            # 5) Initialize new base affine
            new_base = (
                aff_base[:3, :3] * np.abs(aff_base_inv[:3, :3]) * self.inputs.pixel_size
            )

            # 6) Find new center
            new_center_idx = (new_size - 1) * 0.5
            new_affine_base = np.eye(4)
            new_affine_base[:3, :3] = new_base
            new_affine_base[:3, 3] = center_mm[:3] - new_base.dot(new_center_idx)

            # 7) Rotate new matrix
            rotation = nii.affine.dot(aff_base_inv)
            new_affine = rotation.dot(new_affine_base)

            # 8) Generate new reference image
            hdr = nii.header.copy()
            hdr.set_data_shape(new_size)
            nib.Nifti1Image(
                np.zeros(new_size, dtype=nii.get_data_dtype()), new_affine, hdr
            ).to_filename(REF_FILE_NAME)

            out_prefix, ext = op.splitext(op.basename(self.inputs.in_file))
            if ext == ".gz":
                out_prefix, ext2 = op.splitext(out_prefix)
                ext = ext2 + ext

            out_file_name = OUT_FILE_NAME.format(prefix=out_prefix, ext=ext)
            out_file = op.abspath(out_file_name)

            # 9) Resample new image
            ApplyTransforms(
                dimension=3,
                input_image=self.inputs.in_file,
                reference_image=REF_FILE_NAME,
                interpolation="LanczosWindowedSinc",
                transforms=[pkgrf("mriqc", "data/itk_identity.tfm")],
                output_image=out_file,
            ).run()

            self._results["out_file"] = out_file

            if isdefined(self.inputs.in_mask):
                hdr = nii.header.copy()
                hdr.set_data_shape(new_size)
                hdr.set_data_dtype(np.uint8)
                nib.Nifti1Image(
                    np.zeros(new_size, dtype=np.uint8), new_affine, hdr
                ).to_filename(REF_MASK_NAME)

                out_mask_name = OUT_MASK_NAME.format(prefix=out_prefix, ext=ext)
                out_mask = op.abspath(out_mask_name)
                ApplyTransforms(
                    dimension=3,
                    input_image=self.inputs.in_mask,
                    reference_image=REF_MASK_NAME,
                    interpolation="NearestNeighbor",
                    transforms=[pkgrf("mriqc", "data/itk_identity.tfm")],
                    output_image=out_mask,
                ).run()

                self._results["out_mask"] = out_mask

        return runtime
