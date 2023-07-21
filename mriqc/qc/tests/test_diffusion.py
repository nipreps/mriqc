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
import pytest
import numpy as np
import nibabel as nib
from dipy.core.gradients import gradient_table
from dipy.data.fetcher import fetch_sherbrooke_3shell
from dipy.core.gradients import unique_bvals_magnitude, round_bvals
import os.path as op
from ..diffusion import noise_func, get_spike_mask, get_slice_spike_percentage, get_global_spike_percentage

class DiffusionData(object):
    def get_data(self):
        """
        Generate test data
        """
        _, path = fetch_sherbrooke_3shell()
        fnifti = op.join(path, 'HARDI193.nii.gz')
        fnifti, bval, bvec = [op.join(path, f'HARDI193.{ext}') for
                              ext in ["nii.gz", "bval", "bvec"]]
        img = nib.load(fnifti)
        data = img.get_fdata()
        gtab = gradient_table(bval, bvec)
        return data, gtab

    def shelled_data(self):
        data, gtab = self.get_data()
        rounded_bvals = round_bvals(gtab.bvals)
        unique_rounded_bvals = np.unique(rounded_bvals)

        out_data = []
        for u_bv in unique_rounded_bvals:
            this = data[..., np.where(rounded_bvals == u_bv)]
            out_data.append(this)
        return out_data, gtab


@pytest.fixture
def ddata():
    return DiffusionData()


def test_noise_function(ddata):
    data, gtab = ddata.get_data()
    noise_func(data, gtab)


def test_with_shelled_data(ddata):
    shelled_data, gtab = ddata.shelled_data()
    noise_func_for_shelled_data(shelled_data, gtab)