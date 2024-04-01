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

from mriqc.qc.diffusion import spike_percentage


def test_spike_percentage():
    img = np.random.normal(loc=10, scale=1.0, size=(76, 76, 64, 124))
    msk = np.random.randint(0, high=2, size=(76, 76, 64, 124), dtype=bool)
    val = spike_percentage(img, msk, .5)

    assert np.isclose(val['global'], 0.5, rtol=1, atol=1)

    assert np.min(val['slice']) >= 0
    assert np.max(val['slice']) <= 1
    assert len(val['slice']) == img.ndim
