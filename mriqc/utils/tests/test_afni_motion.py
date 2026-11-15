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
"""Conversions between RAS+ world matrices and 3dvolreg's motion files."""

import nibabel as nb
import numpy as np
import pytest
from nitransforms.linear import load

from mriqc.utils.hmc import (
    afni_matrices,
    afni_params,
    apply_afni_transforms,
    ras_to_afni,
    write_afni_motion,
)

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


def _hbn_header():
    header = nb.Nifti1Header()
    header.set_data_shape(HBN_SHAPE)
    header.set_qform(HBN_AFFINE, code=1)
    header.set_sform(HBN_AFFINE, code=1)
    return header


def test_afni_params():
    """The parameters invert the matrices: exactly for afni_matrices', closely for 3dvolreg's."""
    header = _hbn_header()
    matrices = afni_matrices(VOLREG_PARAMS, header, HBN_SHAPE)
    np.testing.assert_allclose(afni_params(matrices, header, HBN_SHAPE), VOLREG_PARAMS, atol=1e-10)
    # 3dvolreg writes six significant digits in its matrices, four decimals in its parameters
    np.testing.assert_allclose(
        afni_params(VOLREG_MATRICES, header, HBN_SHAPE), VOLREG_PARAMS, atol=2e-3
    )


def _rigid(rng, max_deg, max_mm, center):
    """A random rigid RAS+ matrix, rotating about ``center``."""
    from scipy.spatial.transform import Rotation

    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_euler(
        'xyz', rng.uniform(-max_deg, max_deg, 3), True
    ).as_matrix()
    matrix[:3, 3] = center - matrix[:3, :3] @ center + rng.uniform(-max_mm, max_mm, 3)
    return matrix


@pytest.mark.parametrize('qform_code', [1, 2])
def test_ras_to_afni(tmp_path, qform_code):
    """NiTransforms reads the converted files back as the RAS+ matrices (oblique image)."""
    rng = np.random.default_rng(1234)
    shape = (20, 22, 16, 5)
    header = nb.Nifti1Header()
    header.set_data_shape(shape)
    header.set_qform(HBN_AFFINE, code=qform_code)  # 2 > 1: AFNI reads the qform
    header.set_sform(HBN_AFFINE, code=1)
    img = nb.Nifti1Image(np.zeros(shape, dtype='float32'), None, header)
    in_file = tmp_path / 'dwi.nii.gz'
    img.to_filename(in_file)

    center = nb.affines.apply_affine(HBN_AFFINE, (np.array(shape[:3]) - 1) / 2)
    motion = np.stack([_rigid(rng, 3.0, 3.0, center) for _ in range(shape[3])])
    write_afni_motion(
        ras_to_afni(motion, header),
        header,
        shape[:3],
        tmp_path / 'dwi.1D',
        tmp_path / 'dwi.aff12.1D',
        tool='test',
    )

    params = np.loadtxt(tmp_path / 'dwi.1D')
    assert params.shape == (shape[3], 6)
    np.testing.assert_allclose(
        afni_matrices(params, header, shape[:3]).reshape(-1, 12),
        np.loadtxt(tmp_path / 'dwi.aff12.1D'),
        atol=1e-5,
    )
    read = np.asanyarray(
        load(tmp_path / 'dwi.aff12.1D', fmt='afni', reference=in_file, moving=in_file).matrix
    )
    # Within 1e-4 mm on this oblique header: an error of sign or frame would be millimetres
    np.testing.assert_allclose(read[:, :3, :3], motion[:, :3, :3], atol=1e-5)
    np.testing.assert_allclose(read[:, :3, 3], motion[:, :3, 3], atol=2e-4)


def test_apply_afni_transforms(tmp_path):
    """The identity leaves the series as it was."""
    data = np.random.default_rng(0).normal(100, 10, (10, 12, 8, 3)).astype('float32')
    in_file = tmp_path / 'dwi.nii.gz'
    nb.Nifti1Image(data, np.diag([2.0, 2.0, 2.5, 1.0])).to_filename(in_file)
    np.savetxt(tmp_path / 'id.aff12.1D', np.tile(np.eye(4)[:3].ravel(), (3, 1)))

    out_file = apply_afni_transforms(in_file, tmp_path / 'id.aff12.1D', tmp_path / 'out.nii.gz')
    np.testing.assert_allclose(np.asanyarray(nb.load(out_file).dataobj), data, atol=1e-3)


@pytest.mark.parametrize('clip', [False, True])
def test_apply_afni_transforms_clip(tmp_path, clip):
    """Cubic splines overshoot at a sharp edge; clipping keeps the input's range."""
    data = np.zeros((16, 16, 12, 2), dtype='float32')
    data[4:12, 4:12, 3:9] = 1000.0
    in_file = tmp_path / 'dwi.nii.gz'
    nb.Nifti1Image(data, np.diag([2.0, 2.0, 2.5, 1.0])).to_filename(in_file)
    xfms = np.tile(np.eye(4)[:3].ravel(), (2, 1))
    xfms[1, 3] = 0.7  # a subvoxel shift
    np.savetxt(tmp_path / 'shift.aff12.1D', xfms)

    out_file = apply_afni_transforms(
        in_file, tmp_path / 'shift.aff12.1D', tmp_path / 'out.nii.gz', clip=clip
    )
    realigned = np.asanyarray(nb.load(out_file).dataobj)
    in_range = bool(realigned.min() >= 0.0 and realigned.max() <= 1000.0)
    assert in_range is clip


@pytest.mark.parametrize('dtype', ['float32', 'int16'])
@pytest.mark.parametrize('nvols', [1, 5])
def test_apply_afni_transforms_chunks(tmp_path, nvols, dtype):
    """Resampling a few volumes at a time gives the output of resampling all at once."""
    from nitransforms.resampling import apply

    rng = np.random.default_rng(1234)
    shape = (10, 12, 8)
    data = rng.normal(100, 10, shape + (nvols,)).astype('float32')
    in_file = tmp_path / 'dwi.nii.gz'
    img = nb.Nifti1Image(data if nvols > 1 else data[..., 0], HBN_AFFINE)
    img.set_data_dtype(dtype)  # int16: scaled on disk, read as floats (like DWI drift output)
    img.to_filename(in_file)

    center = nb.affines.apply_affine(HBN_AFFINE, (np.array(shape) - 1) / 2)
    motion = np.stack([_rigid(rng, 2.0, 2.0, center) for _ in range(nvols)])
    header = nb.load(in_file).header
    write_afni_motion(
        ras_to_afni(motion, header), header, shape, tmp_path / 'm.1D', tmp_path / 'm.aff12.1D', 't'
    )
    expected = apply(
        load(tmp_path / 'm.aff12.1D', fmt='afni', reference=in_file, moving=in_file),
        in_file,
        dtype_width=4,
        serialize_nvols=2,
        max_concurrent=2,
        mode='reflect',
    )

    # Two volumes per chunk: with five volumes, the last chunk has a single volume
    out_file = apply_afni_transforms(
        in_file,
        tmp_path / 'm.aff12.1D',
        tmp_path / 'out.nii.gz',
        2,
        clip=True,
        chunk_voxels=2 * 960,
    )
    realigned = nb.load(out_file)
    assert realigned.shape == nb.load(in_file).shape
    expected_data = np.asanyarray(expected.dataobj)
    in_data = np.asanyarray(nb.load(in_file).dataobj)
    expected_data = np.clip(expected_data, in_data.min(), in_data.max()).astype(
        expected_data.dtype
    )
    assert realigned.get_data_dtype() == expected.get_data_dtype()
    # Written as the input's type on disk (scaled integers lose precision once written)
    reference = tmp_path / 'expected.nii.gz'
    expected.__class__(expected_data, expected.affine, expected.header).to_filename(reference)
    assert np.array_equal(
        np.asanyarray(realigned.dataobj), np.asanyarray(nb.load(reference).dataobj)
    )
