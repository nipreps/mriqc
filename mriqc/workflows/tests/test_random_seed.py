# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# Copyright 2024 The NiPreps Developers <nipreps@gmail.com>
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
"""Check that ``--random-seed`` reaches the random steps of the anatomical workflow."""

import nibabel as nb
import numpy as np
import pytest

from mriqc import config
from mriqc.testing import mock_config
from mriqc.workflows.anatomical.base import (
    _enhance,
    headmsk_wf,
    init_brain_tissue_segmentation,
)


def test_enhance_seed(tmp_path):
    """The bright outliers are resampled the same way with the same seed."""
    data = np.random.default_rng(0).uniform(100, 200, size=(20, 20, 20)).astype('float32')
    data[10, 10, 10] = 1e4  # an outlier, above the 99.98th percentile
    in_file, wm_tpm = tmp_path / 'in.nii.gz', tmp_path / 'wm.nii.gz'
    nb.Nifti1Image(data, np.eye(4)).to_filename(in_file)
    nb.Nifti1Image(np.ones_like(data), np.eye(4)).to_filename(wm_tpm)

    def enhance(name, seed):
        out_file = _enhance(str(in_file), str(wm_tpm), str(tmp_path / name), seed=seed)
        return np.asanyarray(nb.load(out_file).dataobj)

    assert np.array_equal(enhance('a.nii.gz', 42), enhance('b.nii.gz', 42))
    assert not np.array_equal(enhance('a.nii.gz', 42), enhance('c.nii.gz', 7))


@pytest.mark.parametrize(('ants', 'numpy'), [(None, None), (1234, 5678)])
def test_seeded_nodes(monkeypatch, ants, numpy):
    """With seeds, Enhance gets the NumPy seed and Atropos does not use a random seed."""
    with mock_config():
        monkeypatch.setattr(config.seeds, 'ants', ants)
        monkeypatch.setattr(config.seeds, 'numpy', numpy)
        enhance = headmsk_wf().get_node('Enhance')
        segmentation = init_brain_tissue_segmentation().get_node('segmentation')

    assert enhance.inputs.seed == numpy
    assert segmentation.inputs.use_random_seed is (ants is None)
