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
"""Check the niimath interfaces."""

import shutil

import nibabel as nb
import numpy as np
import pytest

from mriqc.interfaces.niimath import Moco


def test_moco_cmdline(tmp_path, monkeypatch):
    """The options map onto niimath's -moco, with or without writing the series."""
    monkeypatch.chdir(tmp_path)
    for name in ('dwi.nii.gz', 'ref.nii.gz'):
        (tmp_path / name).touch()

    moco = Moco(in_file='dwi.nii.gz', basefile='ref.nii.gz')
    assert moco.cmdline == (
        'niimath dwi.nii.gz -moco -ref ref.nii.gz -1Dfile dwi.1D dwi_moco.nii.gz'
    )
    outputs = moco._list_outputs()
    assert outputs['out_file'] == str(tmp_path / 'dwi_moco.nii.gz')
    assert outputs['oned_matrix_save'] == str(tmp_path / 'dwi.aff12.1D')

    moco = Moco(in_file='dwi.nii.gz', estimate=True)
    assert moco.cmdline == 'niimath dwi.nii.gz -moco -estimate dwi.1D'
    assert 'out_file' not in moco._list_outputs()


@pytest.mark.skipif(shutil.which('niimath') is None, reason='niimath not installed')
@pytest.mark.parametrize('estimate', [False, True])
def test_moco_run(tmp_path, monkeypatch, estimate):
    """A shifted copy of the first volume is realigned onto it."""
    monkeypatch.chdir(tmp_path)
    grid = np.indices((32, 32, 24), dtype='float32')
    centre = np.array([15.5, 15.5, 11.5])[:, None, None, None]
    blob = np.exp(-(((grid - centre) / [[[[6.0]]], [[[4.0]]], [[[3.0]]]]) ** 2).sum(0))
    # The second volume is the first moved by two voxels along the first axis
    data = np.stack([blob, np.roll(blob, 2, axis=0)], axis=-1) * 1000
    affine = np.diag([2.0, 2.0, 2.5, 1.0])
    nb.Nifti1Image(data.astype('float32'), affine).to_filename('bold.nii.gz')

    result = Moco(in_file='bold.nii.gz', estimate=estimate, num_threads=2).run()
    params = np.loadtxt(result.outputs.oned_file)
    assert params.shape == (2, 6)
    # Two voxels of 2 mm along x, which is R-L: dL is the fifth column
    assert np.allclose(np.abs(params[1]), [0, 0, 0, 0, 4.0, 0], atol=0.05)

    matrices = np.loadtxt(result.outputs.oned_matrix_save).reshape(-1, 3, 4)
    assert np.allclose(matrices[0], np.eye(4)[:3])
    assert np.allclose(np.abs(matrices[1, :, 3]), [4.0, 0, 0], atol=0.05)
    assert estimate or nb.load(result.outputs.out_file).shape == data.shape


def _blob(shift=0):
    """A smooth 3D blob, moved by ``shift`` voxels along the first axis."""
    grid = np.indices((32, 32, 24), dtype='float32')
    centre = np.array([15.5 + shift, 15.5, 11.5])[:, None, None, None]
    widths = np.array([6.0, 4.0, 3.0])[:, None, None, None]
    return np.exp(-(((grid - centre) / widths) ** 2).sum(0)).astype('float32') * 1000


@pytest.mark.skipif(shutil.which('niimath') is None, reason='niimath not installed')
@pytest.mark.parametrize('estimate', [False, True])
def test_moco_single_volume(tmp_path, monkeypatch, estimate):
    """A single 3D volume is realigned onto a reference, as 3dvolreg does."""
    monkeypatch.chdir(tmp_path)
    affine = np.diag([2.0, 2.0, 2.5, 1.0])
    nb.Nifti1Image(_blob(), affine).to_filename('ref.nii.gz')
    nb.Nifti1Image(_blob(shift=2), affine).to_filename('b0.nii.gz')

    result = Moco(in_file='b0.nii.gz', basefile='ref.nii.gz', estimate=estimate).run()
    params = np.loadtxt(result.outputs.oned_file, ndmin=2)
    assert params.shape == (1, 6)
    assert np.allclose(np.abs(params[0]), [0, 0, 0, 0, 4.0, 0], atol=0.05)
    assert np.loadtxt(result.outputs.oned_matrix_save).shape == (12,)
    assert estimate or nb.load(result.outputs.out_file).shape == (32, 32, 24)


@pytest.mark.skipif(shutil.which('niimath') is None, reason='niimath not installed')
def test_moco_error(tmp_path, monkeypatch):
    """niimath's own error is reported (here, a reference on another grid)."""
    monkeypatch.chdir(tmp_path)
    nb.Nifti1Image(np.stack([_blob(), _blob()], axis=-1), np.eye(4)).to_filename('bold.nii.gz')
    nb.Nifti1Image(_blob()[:16], np.eye(4)).to_filename('ref.nii.gz')

    with pytest.raises(RuntimeError, match='Return code: 1'):
        Moco(in_file='bold.nii.gz', basefile='ref.nii.gz').run()
