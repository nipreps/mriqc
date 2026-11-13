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
"""Resampling of the BOLD series with the head-motion transforms."""

import nibabel as nb
import numpy as np
from nitransforms.linear import load
from nitransforms.resampling import apply

from mriqc.workflows.functional.base import _apply_transforms


def test_apply_transforms_in_memory(tmp_path, monkeypatch):
    """Resampling the series loaded in memory gives the output of reading it from disk."""
    rng = np.random.default_rng(1234)
    nvols = 6
    in_file = tmp_path / 'sub-01_task-rest_bold.nii.gz'
    data = rng.normal(100, 10, (10, 12, 8, nvols)).astype('float32')
    nb.Nifti1Image(data, np.diag([3.0, 3.0, 3.3, 1.0])).to_filename(in_file)

    # 3dvolreg's -1Dmatrix_save format: one row per volume with a 3x4 matrix
    xfms = np.tile(np.eye(4)[:3].ravel(), (nvols, 1))
    xfms[:, 3] = np.linspace(0, 0.6, nvols)
    xfms[:, 7] = np.linspace(0, -0.4, nvols)
    xfm_file = tmp_path / 'hmc.aff12.1D'
    np.savetxt(xfm_file, xfms)

    expected = apply(
        load(xfm_file, fmt='afni', reference=in_file, moving=in_file),
        in_file,
        dtype_width=4,
        serialize_nvols=2,
        max_concurrent=2,
        mode='reflect',
    )

    monkeypatch.chdir(tmp_path)
    out_file = _apply_transforms(str(in_file), str(xfm_file), 2)
    realigned = np.asanyarray(nb.load(out_file).dataobj)
    assert realigned.shape == data.shape
    assert not np.array_equal(realigned, data)  # the transforms moved the volumes
    assert np.array_equal(realigned, np.asanyarray(expected.dataobj))
