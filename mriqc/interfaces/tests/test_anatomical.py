# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# Copyright 2026 The NiPreps Developers <nipreps@gmail.com>
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
"""Tests for the anatomical interfaces."""

import nibabel as nb
import numpy as np
import pytest

from mriqc.interfaces.anatomical import Harmonize


def _write(path, data):
    nb.Nifti1Image(data.astype('float32'), np.eye(4)).to_filename(path)
    return str(path)


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    data = np.random.default_rng(1234).uniform(100, 200, size=(20, 20, 20))
    brain = np.zeros(data.shape)
    brain[2:-2, 2:-2, 2:-2] = 1
    return {
        'in_file': _write(tmp_path / 'in.nii.gz', data),
        'brain_mask': _write(tmp_path / 'brain.nii.gz', brain),
    }


def _harmonize(tmp_path, inputs, wm):
    wm_mask = _write(tmp_path / 'wm.nii.gz', wm)
    out_file = Harmonize(wm_mask=wm_mask, **inputs).run().outputs.out_file
    return nb.load(inputs['in_file']).get_fdata(), nb.load(out_file).get_fdata()


def test_harmonize(tmp_path, inputs):
    """The median within the eroded white-matter mask becomes 1000."""
    wm = np.zeros((20, 20, 20))
    wm[5:15, 5:15, 5:15] = 1.0
    _, out_data = _harmonize(tmp_path, inputs, wm)

    eroded = np.zeros(wm.shape, dtype=bool)
    eroded[6:14, 6:14, 6:14] = True
    assert np.isclose(np.median(out_data[eroded]), 1000, rtol=1e-4)


def test_harmonize_sparse_wm(tmp_path, inputs):
    """A white-matter mask that erosion empties falls back to the brain mask."""
    wm = np.zeros((20, 20, 20))
    wm[2:-2:3, 2:-2:3, 2:-2:3] = 1.0  # isolated voxels, which erosion removes
    assert wm.sum() >= 30  # large enough before erosion

    data, out_data = _harmonize(tmp_path, inputs, wm)

    assert np.all(np.isfinite(out_data))
    brain = nb.load(inputs['brain_mask']).get_fdata() > 0.5
    lo, hi = np.percentile(data[brain], [75, 95])
    fallback = brain & (data >= lo) & (data <= hi)
    assert np.isclose(np.median(out_data[fallback]), 1000, rtol=1e-4)
