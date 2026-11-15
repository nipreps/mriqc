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
"""Tests for the DWI interfaces."""

from pathlib import Path

import nibabel as nb
import numpy as np
import pytest

from mriqc.interfaces.diffusion import PIESNO, DWISummary

B_DICT = {0: [0, 9], 1000: [1, 2, 3, 4], 2000: [5, 6, 7, 8]}


def _summary(tmp_path, monkeypatch, **inputs):
    monkeypatch.chdir(tmp_path)
    in_file = tmp_path / 'sub-01_dwi.nii.gz'
    in_file.write_bytes(b'')
    inputs = {'n_shells': 2, 'is_dsi': False, 'b_dict': B_DICT} | inputs
    result = DWISummary(in_file=str(in_file), **inputs).run()
    return Path(result.outputs.out_report).read_text()


def test_dwi_summary(tmp_path, monkeypatch):
    """The summary renders the metadata and shells it is given."""
    html = _summary(tmp_path, monkeypatch, pe_direction='j-', echo_time=0.089)

    assert '<li>Filename: sub-01_dwi.nii.gz</li>' in html
    assert '<li>Phase-encoding (PE) direction: j-</li>' in html
    assert '<li>Echo time (TE): 89 ms</li>' in html
    assert '<li>Repetition time (TR): n/a</li>' in html
    assert '<li>Number of b=0: 2</li>' in html
    assert '<li>Indices of the b=0 scans: 0, 9</li>' in html
    assert '<li>Type of DWIs: multi-shell</li>' in html
    assert '<li>Number of DWIs: 4 at b=1000, 4 at b=2000</li>' in html


@pytest.mark.parametrize(
    ('n_shells', 'is_dsi', 'dwi_type'),
    [(1, False, 'single-shell'), (2, False, 'multi-shell'), (3, True, 'DSI')],
)
def test_dwi_summary_type(tmp_path, monkeypatch, n_shells, is_dsi, dwi_type):
    """The type of DWI follows the shells found upstream."""
    html = _summary(tmp_path, monkeypatch, n_shells=n_shells, is_dsi=is_dsi)
    assert f'<li>Type of DWIs: {dwi_type}</li>' in html


def test_piesno(tmp_path, monkeypatch):
    """PIESNO's sigma ignores mostly zero-valued slices, without dividing by zero."""
    monkeypatch.chdir(tmp_path)
    rng = np.random.default_rng(1234)
    shape = (16, 16, 6, 10)
    # Rician noise (single channel) with a sigma of 10
    noise = np.abs(rng.normal(0, 10.0, shape) + 1j * rng.normal(0, 10.0, shape))
    # Most slices are zero-valued: they would drag the median sigma down to zero
    data = noise.copy()
    data[:, :, :4] = 0
    # A slice with a single nonzero volume makes PIESNO divide by zero (gh-1333)
    data[:, :, 0, 0] = noise[:, :, 0, 0]
    in_file = tmp_path / 'sub-01_dwi.nii.gz'
    nb.Nifti1Image(data.astype('float32'), np.eye(4)).to_filename(in_file)

    sigma = PIESNO(in_file=str(in_file), n_channels=1).run().outputs.sigma

    assert sigma == pytest.approx(10.0, rel=0.05)


def test_rotate_vectors_affine(tmp_path, monkeypatch):
    """Affine transforms turn the b-vectors by their rotation, not by their stretch."""
    from scipy.spatial.transform import Rotation

    from mriqc.interfaces.diffusion import RotateVectors

    monkeypatch.chdir(tmp_path)
    nb.Nifti1Image(
        np.zeros((8, 8, 8), dtype='float32'), np.diag([2.0, 2.0, 2.5, 1.0])
    ).to_filename('ref.nii.gz')
    np.savetxt('dwi.bvec', np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0.6, 0, 0.8]]).T)

    rotation = Rotation.from_euler('xyz', [2.0, -1.0, 3.0], degrees=True).as_matrix()
    stretch = np.array([[1.01, 0.01, 0.0], [0.01, 1.03, 0.005], [0.0, 0.005, 0.99]])
    results = {}
    for name, linear in (('rigid', rotation), ('affine', rotation @ stretch)):
        matrices = np.tile(np.eye(4)[:3], (4, 1, 1))
        matrices[1:, :, :3] = linear
        np.savetxt(f'{name}.aff12.1D', matrices.reshape(-1, 12))
        results[name] = (
            RotateVectors(
                in_file='dwi.bvec', reference='ref.nii.gz', transforms=f'{name}.aff12.1D'
            )
            .run()
            .outputs
        )

    np.testing.assert_allclose(results['affine'].out_diff, results['rigid'].out_diff, atol=2e-6)
    np.testing.assert_allclose(results['affine'].out_bvec, results['rigid'].out_bvec, atol=1e-6)
    assert results['rigid'].out_diff[0] == 0.0  # b=0
    assert all(0.01 < diff < 0.07 for diff in results['rigid'].out_diff[1:])  # 0.6 to 4 degrees
