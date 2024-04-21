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

from pathlib import Path

import numpy as np

from mriqc.qc.diffusion import plot_gradients, spike_ppm


def test_spike_ppm():
    msk = np.random.randint(0, high=2, size=(76, 76, 64, 124), dtype=bool)
    val = spike_ppm(msk, 0.5)

    assert np.isclose(val['global'], 0.5e6, rtol=1, atol=1)

    assert np.min([val[f'slice_{ax}'] for ax in 'ijk']) >= 0
    assert np.max([val[f'slice_{ax}'] for ax in 'ijk']) <= 1e6
    assert len([val[f'slice_{ax}'] for ax in 'ijk']) == msk.ndim - 1


def test_plot_gradients(tmp_path):
    from dipy.io import read_bvals_bvecs
    from dipy.core.gradients import gradient_table
    fbval = "./mriqc/data/testdata/hcp_bvals"
    fbvec = "./mriqc/data/testdata/hcp_bvecs"
    _bvals, _bvecs = read_bvals_bvecs(fbval, fbvec)
    gtab = gradient_table(_bvals, _bvecs)
    bvecs = gtab.bvecs[~gtab.b0s_mask]
    bvals = gtab.bvals[~gtab.b0s_mask]

    gradients = np.vstack([bvecs.T, bvals])
    _ = plot_gradients(gradients)

    from matplotlib import pyplot as plt
    plt.savefig(Path(tmp_path) / "gradients.png")
