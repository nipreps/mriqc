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
from statsmodels.robust.scale import mad

from mriqc.qc.diffusion import cc_snr, neighboring_dwi_correlation, spike_ppm


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


def _cc_data():
    """Generate a cubic corpus callosum mask and a noisy b=0 image."""
    rng = np.random.default_rng(42)
    cc_mask = np.zeros((10, 10, 10), dtype=bool)
    cc_mask[3:7, 3:7, 3:7] = True
    return rng, cc_mask, rng.normal(1000.0, 50.0, size=cc_mask.shape)


def _shell(cc_mask, values):
    """Generate a shell whose volumes are constant (one value each) within the CC."""
    data = np.zeros((*cc_mask.shape, len(values)))
    data[cc_mask] = values
    return data


def test_cc_snr():
    """Each diffusion-weighted shell reports SNR on its own data and b-vectors."""
    _, cc_mask, in_b0 = _cc_data()
    sigma = mad(in_b0[cc_mask])

    # The CC runs left-right, so the volume closest to X is the darkest of each shell
    shells = [_shell(cc_mask, [600.0, 200.0, 600.0]), _shell(cc_mask, [150.0, 50.0, 150.0])]
    bvecs = [
        [(0.0, 1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)],
        # -X is as aligned with X as +X
        [(0.0, 0.8, 0.6), (-1.0, 0.0, 0.0), (0.0, 0.0, 1.0)],
    ]

    snr, cc_sigma = cc_snr(in_b0, shells, cc_mask, [1000, 2000], bvecs)

    assert cc_sigma == sigma
    # gh-1358: best and worst must differ when the volume closest to X is darker
    assert snr['shell1_best'] != snr['shell1_worst']
    assert set(snr) == {'shell0', 'shell1_worst', 'shell1_best', 'shell2_worst', 'shell2_best'}
    assert snr['shell0'] == round(in_b0[cc_mask].mean() / sigma, 2)
    assert snr['shell1_worst'] == round(200.0 / sigma, 2)
    assert snr['shell1_best'] == round(600.0 / sigma, 2)
    assert snr['shell2_worst'] == round(50.0 / sigma, 2)
    assert snr['shell2_best'] == round(150.0 / sigma, 2)


def test_cc_snr_single_volume_shell():
    """A shell with a single volume, squeezed into 3D, is still averaged within the CC."""
    rng, cc_mask, in_b0 = _cc_data()
    shell = rng.normal(400.0, 50.0, size=cc_mask.shape)

    snr, cc_sigma = cc_snr(in_b0, [shell], cc_mask, [1000], [[(1.0, 0.0, 0.0)]])

    expected = round(shell[cc_mask].mean() / cc_sigma, 2)
    assert snr['shell1_worst'] == snr['shell1_best'] == expected
