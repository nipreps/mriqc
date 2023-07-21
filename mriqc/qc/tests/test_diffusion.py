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
import nibabel as nib
from dipy.core.gradients import gradient_table
from dipy.data.fetcher import fetch_sherbrooke_3shell
import os.path as op
from ..diffusion import noise_func, get_spike_mask, get_slice_spike_percentage, get_global_spike_percentage
import numpy as np

z_threshold = 2
slice_threshold = .2


def test_get_spike_mask(ddata):
    img, gtab = ddata.get_fdata()
    spike_mask = get_spike_mask(img, 2)

    assert np.min(np.ravel(spike_mask)) == 0
    assert np.max(np.ravel(spike_mask)) == 1
    assert spike_mask.shape == img.shape


def test_get_slice_spike_percentage(ddata):
    img, gtab = ddata.get_fdata()
    slice_spike_percentage = get_slice_spike_percentage(img, 2, .2)

    assert np.min(slice_spike_percentage) >= 0
    assert np.max(slice_spike_percentage) <= 1
    assert len(slice_spike_percentage) == img.ndim


def test_get_global_spike_percentage(ddata):
    img, gtab = ddata.get_fdata()
    global_spike_percentage = get_global_spike_percentage(img, 2)

    assert global_spike_percentage >= 0
    assert global_spike_percentage <= 1
