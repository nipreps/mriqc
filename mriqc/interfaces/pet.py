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
import os

import nibabel as nb
import numpy as np
import pandas as pd
from nipype.interfaces.base import (
    BaseInterfaceInputSpec,
    File,
    TraitedSpec
)

class ChooseRefHMCInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True, desc="Input PET image file")

class ChooseRefHMCOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc="Output file with the selected reference frame")

class ChooseRefHMC(BaseInterface):
    input_spec = ChooseRefHMCInputSpec
    output_spec = ChooseRefHMCOutputSpec

    def _run_interface(self, runtime):
        in_file = self.inputs.in_file

        # Load the PET image
        img = nb.load(in_file)
        data = img.get_fdata()

        # Compute the sum of intensities across voxels for each time frame
        frame_sums = np.sum(data, axis=(0, 1, 2))

        # Find the time frame with the highest sum of intensity
        max_frame_idx = np.argmax(frame_sums)

        # Extract the corresponding frame
        max_frame_data = data[..., max_frame_idx]

        # Create a new NIfTI image for the selected frame
        max_frame_img = nb.Nifti1Image(max_frame_data, img.affine, img.header)

        # Save the new NIfTI image
        output_filename = os.path.abspath("max_intensity_frame.nii.gz")
        nb.save(max_frame_img, output_filename)

        self._results["out_file"] = output_filename
        return runtime

class FDStatsInputSpec(BaseInterfaceInputSpec):
    in_fd = File(exists=True, mandatory=True, desc="Input FD file")
    fd_thres = File(mandatory=True, desc="FD threshold value")

class FDStatsOutputSpec(TraitedSpec):
    out_fd = dict(desc="Dictionary with FD metrics: mean, num, perc")

class FDStats(BaseInterface):
    input_spec = FDStatsInputSpec
    output_spec = FDStatsOutputSpec

    def _run_interface(self, runtime):
        # Load FD data
        fd_data = np.loadtxt(self.inputs.in_fd, skiprows=1)

        # Compute number of FD values above the threshold
        num_fd = (fd_data > self.inputs.fd_thres).sum()

        # Store results in the output dictionary
        self._results['out_fd'] = {
            'mean': float(fd_data.mean()),
            'num': int(num_fd),
            'perc': float(num_fd * 100 / (len(fd_data) + 1)),
        }

        return runtime
