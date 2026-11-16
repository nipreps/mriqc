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
import importlib.util
import shutil
import sys
import types

import nibabel as nb
import numpy as np
import pytest

from mriqc.interfaces.nifreeze import NiFreezeHMC
from mriqc.utils.hmc import afni_params, ras_to_afni

AFFINE = np.diag([2.0, 2.0, 2.5, 1.0])


def _blob(shift=0):
    grid = np.indices((32, 32, 24), dtype='float32')
    centre = np.array([15.5, 15.5, 11.5])[:, None, None, None]
    widths = np.array([6.0, 4.0, 3.0])[:, None, None, None]
    blob = np.exp(-(((grid - centre) / widths) ** 2).sum(0)) * 1000
    return np.roll(blob, shift, axis=0).astype('float32')


def test_single_b0(tmp_path, monkeypatch):
    """A single b=0 is the reference: it does not move, and NiFreeze is not needed."""
    monkeypatch.chdir(tmp_path)
    nb.Nifti1Image(_blob(), AFFINE).to_filename('ref.nii.gz')
    nb.Nifti1Image(_blob()[..., None], AFFINE).to_filename('b0.nii.gz')

    result = NiFreezeHMC(in_file='b0.nii.gz', basefile='ref.nii.gz').run()
    assert np.array_equal(np.loadtxt(result.outputs.oned_file, ndmin=2), np.zeros((1, 6)))
    assert np.allclose(np.loadtxt(result.outputs.oned_matrix_save), np.eye(4)[:3].ravel())
    assert np.array_equal(nb.load(result.outputs.out_file).get_fdata()[..., 0], _blob())


def test_bvecs_required(tmp_path, monkeypatch):
    """Diffusion-weighted volumes cannot be predicted without their b-vectors."""
    monkeypatch.chdir(tmp_path)
    nb.Nifti1Image(np.stack([_blob()] * 2, axis=-1), AFFINE).to_filename('dwi.nii.gz')
    nb.Nifti1Image(_blob(), AFFINE).to_filename('ref.nii.gz')
    np.savetxt('dwi.bval', [0, 1000])

    with pytest.raises(ValueError, match='b-vectors'):
        NiFreezeHMC(in_file='dwi.nii.gz', basefile='ref.nii.gz', in_bval='dwi.bval').run()


@pytest.mark.skipif(importlib.util.find_spec('nifreeze') is None, reason='nifreeze not installed')
@pytest.mark.skipif(shutil.which('antsRegistration') is None, reason='ANTs not installed')
@pytest.mark.parametrize('estimate', [False, True])
def test_b0s(tmp_path, monkeypatch, estimate):
    """Several b=0 volumes are registered onto the reference, in 3dvolreg's convention."""
    monkeypatch.chdir(tmp_path)
    nb.Nifti1Image(_blob(), AFFINE).to_filename('ref.nii.gz')
    # The second volume is moved two voxels (4 mm) to the right (+x, RAS)
    data = np.stack([_blob(), _blob(shift=2), _blob()], axis=-1)
    nb.Nifti1Image(data, AFFINE).to_filename('b0.nii.gz')

    result = NiFreezeHMC(
        in_file='b0.nii.gz', basefile='ref.nii.gz', seed=42, estimate=estimate
    ).run()
    params = np.loadtxt(result.outputs.oned_file, ndmin=2)
    assert params.shape == (3, 6)
    # As 3dvolreg reports it: dL = +4 (see test_afni_matrices for the convention). ANTs
    # finds rotations of up to 0.5 degrees on this smooth blob.
    np.testing.assert_allclose(params[1, 3:], [0, 4.0, 0], atol=0.1)
    np.testing.assert_allclose(params[[0, 2], 3:], 0, atol=0.1)
    np.testing.assert_allclose(params[:, :3], 0, atol=1.0)
    assert np.loadtxt(result.outputs.oned_matrix_save).shape == (3, 12)
    if estimate:
        assert not result.outputs.out_file
    else:
        realigned = nb.load(result.outputs.out_file).get_fdata()
        assert np.abs(realigned[..., 1] - _blob()).max() < 0.05 * _blob().max()


def _fake_nifreeze(monkeypatch, passes):
    """Stand in for NiFreeze: each pass moves the volumes, and is recorded in ``passes``."""
    shift = np.eye(4)
    shift[:3, 3] = [1.0, -0.5, 0.25]  # the rigid pass: a translation
    stretch = np.eye(4)
    stretch[1, 1], stretch[0, 1] = 1.02, 0.01  # the affine pass: scaling and shear along y

    class Estimator:
        def __init__(self, model, **kwargs):
            self.model, self.kwargs = model, kwargs

        def run(self, dataset, **kwargs):
            passes.append((self.model, self.kwargs, kwargs, dataset.motion_affines.copy()))
            affine = self.kwargs.get('ants_config') == 'dwi-to-dwi_level1'
            # In place, as NiFreeze's set_transform does
            dataset.motion_affines[:] = dataset.motion_affines @ (stretch if affine else shift)
            return self

    def from_nii(in_file, brainmask_file, bvec_file, bval_file, b0_file):
        n = int((np.loadtxt(bval_file) > 50).sum())
        return types.SimpleNamespace(motion_affines=np.tile(np.eye(4), (n, 1, 1)))

    modules = {name: types.ModuleType(name) for name in ('nifreeze', 'nifreeze.data')}
    modules['nifreeze.estimator'] = types.ModuleType('nifreeze.estimator')
    modules['nifreeze.estimator'].Estimator = Estimator
    modules['nifreeze.data.dmri'] = types.ModuleType('nifreeze.data.dmri')
    modules['nifreeze.data.dmri'].from_nii = from_nii
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    return shift, shift @ stretch


@pytest.mark.parametrize('eddy', [False, True])
def test_eddy(tmp_path, monkeypatch, eddy):
    """With eddy, an affine pass follows the rigid one: FD keeps the rigid parameters."""
    monkeypatch.chdir(tmp_path)
    passes = []
    rigid, affine = _fake_nifreeze(monkeypatch, passes)
    nb.Nifti1Image(_blob(), AFFINE).to_filename('ref.nii.gz')
    nb.Nifti1Image(np.stack([_blob()] * 3, axis=-1), AFFINE).to_filename('dwi.nii.gz')
    np.savetxt('dwi.bval', [0, 1000, 2000])
    np.savetxt('dwi.bvec', [[0, 1, 0], [0, 0, 1], [0, 0, 0]])

    result = NiFreezeHMC(
        in_file='dwi.nii.gz',
        basefile='ref.nii.gz',
        in_bvec='dwi.bvec',
        in_bval='dwi.bval',
        seed=42,
        num_threads=2,
        eddy=eddy,
    ).run()

    assert [p[1] for p in passes] == [{}, {'ants_config': 'dwi-to-dwi_level1'}][: 1 + eddy]
    assert all(p[0] == 'gqi-krr' and p[2] == {'omp_nthreads': 2, 'n_jobs': 2, 'seed': 42}
               for p in passes)  # fmt: skip
    if eddy:  # the affine pass starts from the rigid estimates
        np.testing.assert_array_equal(passes[1][3], np.stack([rigid] * 2))

    # The single b=0 is the reference; the diffusion-weighted volumes moved
    header = nb.load('dwi.nii.gz').header
    expected = np.stack([np.eye(4), *[affine if eddy else rigid] * 2])
    np.testing.assert_allclose(
        np.loadtxt(result.outputs.oned_matrix_save),
        ras_to_afni(expected, header).reshape(-1, 12),
        atol=1e-6,
    )
    params = afni_params(
        ras_to_afni(np.stack([np.eye(4), rigid, rigid]), header), header, (32, 32, 24)
    )
    np.testing.assert_allclose(np.loadtxt(result.outputs.oned_file), params, atol=1e-6)
    assert nb.load(result.outputs.out_file).shape == (32, 32, 24, 3)
