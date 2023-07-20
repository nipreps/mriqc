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
import numpy as np
from mriqc.mriqc.qc import get_spike_mask, get_slice_spike_percentage, get_global_spike_percentage


slice = np.array([[1, 1, 10, 1], [1, 10, 1, 1], [1, 10, 1, 1], [1, 1, 1, 1]])
test_data = np.array([slice, slice, slice, slice, slice])

z_threshold = 2
slice_threshold = .2

def test_get_spike_mask():
    spike_mask = get_spike_mask(test_data, z_threshold)

    assert np.min(np.ravel(spike_mask)) == 0
    assert np.max(np.ravel(spike_mask)) == 1
    assert spike_mask.shape == test_data.shape


def test_get_slice_spike_percentage():
    slice_spike_percentage = get_slice_spike_percentage(test_data, z_threshold, slice_threshold)

    assert np.min(np.ravel(slice_spike_percentage)) >= 0
    assert np.max(np.ravel(slice_spike_percentage)) <= 1
    assert len(slice_spike_percentage) == test_data.ndim


def test_get_global_spike_percentage():
    global_spike_percentage = get_global_spike_percentage(test_data, z_threshold)

    assert np.min(np.ravel(global_spike_percentage)) >= 0
    assert np.max(np.ravel(global_spike_percentage)) <= 1
