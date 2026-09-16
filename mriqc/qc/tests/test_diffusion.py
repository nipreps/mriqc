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

from mriqc.qc.diffusion import neighboring_dwi_correlation, spike_ppm


def test_spike_ppm():
    msk = np.random.randint(0, high=2, size=(76, 76, 64, 124), dtype=bool)
    val = spike_ppm(msk, 0.5)

    assert np.isclose(val['global'], 0.5e6, rtol=1, atol=1)

    assert np.min([val[f'slice_{ax}'] for ax in 'ijk']) >= 0
    assert np.max([val[f'slice_{ax}'] for ax in 'ijk']) <= 1e6
    assert len([val[f'slice_{ax}'] for ax in 'ijk']) == msk.ndim - 1


def test_neighboring_dwi_correlation():
    rng = np.random.default_rng(42)
    data = rng.random((5, 5, 5, 4))
    # Volume 1 is a copy of volume 0, so their correlation is exactly 1
    data[..., 1] = data[..., 0]

    assert neighboring_dwi_correlation(data, [(0, 1)]) == 1.0

    mask = np.zeros(data.shape[:-1], dtype=bool)
    mask[1:4, 1:4, 1:4] = True
    assert neighboring_dwi_correlation(data, [(0, 1)], mask=mask) == 1.0
