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

from mriqc.interfaces.niimath import Moco, afni_matrices

# Three volumes of an HBN DWI run (oblique), realigned by 3dvolreg (AFNI 26.0.07):
# its -1Dfile parameters and its -1Dmatrix_save matrices.
HBN_AFFINE = np.array(
    [
        [-1.79009676, 0.00000009, 0.16939613, 90.46134186],
        [-0.01180378, 1.79369688, -0.12500422, -83.0014801],
        [0.1688032, 0.1254275, 1.78764367, -94.22790527],
        [0.0, 0.0, 0.0, 1.0],
    ]
)
HBN_SHAPE = (104, 104, 72)
VOLREG_PARAMS = np.array(
    [
        [0.5071, 7.0387, -0.0961, 8.3341, 0.4766, -2.6787],
        [0.1705, 7.0124, -0.5696, 8.4216, 0.5818, -3.0870],
        [0.7173, 8.3701, 6.7830, 4.6742, 1.1331, -2.6984],
    ]
)
VOLREG_MATRICES = np.array(
    [
        [0.999958, 0.00878402, 0.00276198, -0.307877, -0.00905626, 0.992425,
         0.12252, 5.30405, -0.00166483, -0.12254, 0.992462, -10.0001],
        [0.999943, 0.00295412, 0.0103048, -0.318432, -0.00418993, 0.992515,
         0.122048, 5.67706, -0.00986716, -0.122084, 0.992471, -10.1085],
        [0.993138, 0.0123849, -0.116291, -3.94159, 0.00476081, 0.989271,
         0.146015, 6.2967, 0.116852, -0.145566, 0.982424, -7.29765],
    ]
).reshape(-1, 3, 4)  # fmt: skip


def test_afni_matrices():
    """The matrices match 3dvolreg's, to the precision of its parameter file."""
    header = nb.Nifti1Header()
    header.set_data_shape(HBN_SHAPE)
    header.set_qform(HBN_AFFINE, code=1)
    header.set_sform(HBN_AFFINE, code=1)

    matrices = afni_matrices(VOLREG_PARAMS, header, HBN_SHAPE)
    np.testing.assert_allclose(matrices[..., :3], VOLREG_MATRICES[..., :3], atol=1e-5)
    np.testing.assert_allclose(matrices[..., 3], VOLREG_MATRICES[..., 3], atol=2e-4)


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
