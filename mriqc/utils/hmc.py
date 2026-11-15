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
"""
Head-motion estimates in AFNI 3dvolreg's convention, whatever the tool that made them.

MRIQC computes framewise displacement from 3dvolreg's ``-1Dfile`` parameters, and rotates
b-vectors and resamples series with its ``-1Dmatrix_save`` matrices. These helpers convert
between those two files and RAS+ world matrices.

"""

from itertools import permutations
from pathlib import Path

import numpy as np


def cardinal_affine(header):
    """
    Return 3dvolreg's voxel-to-DICOM matrix, and the image's own affine.

    3dvolreg and niimath ignore obliquity: they snap each voxel axis onto the nearest
    DICOM axis (+x Left, +y Posterior, +z Superior). They read the sform, unless the
    qform has a higher code.

    """
    sform, sform_code = header.get_sform(coded=True)
    qform, qform_code = header.get_qform(coded=True)
    affine = sform if sform_code > 0 and not sform_code < qform_code else qform
    zooms = np.abs(np.array(header.get_zooms()[:3], dtype=float))
    zooms[zooms <= 0] = 1.0

    cardinal = np.diag([*zooms, 1.0])
    if affine is not None:
        flip = np.array([-1.0, -1.0, 1.0])  # RAS+ to DICOM
        linear = flip[:, None] * affine[:3, :3]
        best = max(
            permutations(range(3)),
            key=lambda perm: sum(abs(linear[perm[j], j]) for j in range(3)),
        )
        cardinal[:3, :3] = 0.0
        for j, i in enumerate(best):
            cardinal[i, j] = np.copysign(zooms[j], linear[i, j])
        cardinal[:3, 3] = flip * affine[:3, 3]
    return cardinal, affine


def _center(header, shape):
    """Centre of the cardinal grid (DICOM mm), about which 3dvolreg rotates."""
    cardinal, _ = cardinal_affine(header)
    return cardinal[:3, :3] @ ((np.array(shape[:3]) - 1) / 2) + cardinal[:3, 3]


def afni_matrices(params, header, shape):
    """
    Compute the matrices of 3dvolreg's ``-1Dmatrix_save`` from its motion parameters.

    Each matrix maps a point of the base (DICOM mm, on the cardinal grid of
    :func:`cardinal_affine`) onto the input volume, rotating about the grid's centre.

    Parameters
    ----------
    params : :obj:`numpy.ndarray`
        One row per volume: roll, pitch, yaw (degrees), dS, dL, dP (mm).
    header : :obj:`nibabel.nifti1.Nifti1Header`
        Header of the realigned series.
    shape : :obj:`tuple`
        Its three spatial dimensions.

    Returns
    -------
    :obj:`numpy.ndarray`
        One 3x4 matrix per volume.

    """
    center = _center(header, shape)
    params = np.asanyarray(params, dtype=float)
    roll, pitch, yaw = np.deg2rad(params[:, :3]).T
    rotations = _rotation(2, -roll) @ _rotation(0, -pitch) @ _rotation(1, -yaw)
    shifts = params[:, [4, 5, 3]]  # (dL, dP, dS) = DICOM (x, y, z)
    translations = -np.einsum('nij,nj->ni', rotations, shifts) + center - rotations @ center
    return np.concatenate((rotations, translations[..., None]), axis=-1)


def afni_params(matrices, header, shape):
    """
    Compute 3dvolreg's motion parameters from its matrices: the inverse of :func:`afni_matrices`.

    Parameters
    ----------
    matrices : :obj:`numpy.ndarray`
        One rigid 3x4 matrix per volume, as 3dvolreg's ``-1Dmatrix_save`` writes them.
    header : :obj:`nibabel.nifti1.Nifti1Header`
        Header of the realigned series.
    shape : :obj:`tuple`
        Its three spatial dimensions.

    Returns
    -------
    :obj:`numpy.ndarray`
        One row per volume: roll, pitch, yaw (degrees), dS, dL, dP (mm).

    """
    center = _center(header, shape)
    matrices = np.asanyarray(matrices, dtype=float)
    rotations, translations = matrices[..., :3], matrices[..., 3]
    # rotations = Rz(-roll) Rx(-pitch) Ry(-yaw)
    pitch = -np.arcsin(np.clip(rotations[:, 2, 1], -1.0, 1.0))
    yaw = -np.arctan2(-rotations[:, 2, 0], rotations[:, 2, 2])
    roll = -np.arctan2(-rotations[:, 0, 1], rotations[:, 1, 1])
    # R^T (c - t) - c = (dL, dP, dS)
    shifts = np.einsum('nji,nj->ni', rotations, center - translations) - center
    return np.column_stack((np.rad2deg(np.column_stack((roll, pitch, yaw))), shifts[:, [2, 0, 1]]))


def ras_to_afni(matrices, header, affine=None):
    """
    Express RAS+ world matrices (reference to moving) in 3dvolreg's convention.

    The motion is the same in voxel space, :math:`A^{-1} M A` with the affine
    :math:`A` the matrices were estimated in, and 3dvolreg places those voxels on its
    cardinal grid :math:`C`, so its matrix is :math:`C A^{-1} M A C^{-1}`.

    Parameters
    ----------
    matrices : :obj:`numpy.ndarray`
        One 4x4 RAS+ matrix per volume, mapping reference coordinates onto the volume.
    header : :obj:`nibabel.nifti1.Nifti1Header`
        Header of the series.
    affine : :obj:`numpy.ndarray`, optional
        The affine the matrices were estimated in (default: nibabel's, as NiBabel-based
        tools read it).

    Returns
    -------
    :obj:`numpy.ndarray`
        One 3x4 matrix per volume, as 3dvolreg's ``-1Dmatrix_save`` writes them.

    """
    cardinal, _ = cardinal_affine(header)
    if affine is None:
        affine = header.get_best_affine()
    to_cardinal = cardinal @ np.linalg.inv(affine)
    return (to_cardinal @ np.asanyarray(matrices) @ np.linalg.inv(to_cardinal))[:, :3, :]


def write_afni_motion(matrices, header, shape, oned_file, oned_matrix_save, tool, rigid=None):
    """
    Write 3dvolreg's ``-1Dfile`` and ``-1Dmatrix_save`` files from its (N, 3, 4) matrices.

    The parameters are those of ``rigid`` when given: affine matrices (e.g., with
    eddy-current correction) have no motion parameters, so they come from the rigid
    estimates of the same volumes.

    """
    rigid = matrices if rigid is None else rigid
    np.savetxt(oned_file, afni_params(rigid, header, shape), fmt='%.6f')
    np.savetxt(
        oned_matrix_save,
        matrices.reshape(-1, 12),
        fmt='%.8g',
        header=f'3dvolreg matrices (DICOM-to-DICOM, row-by-row), from {tool}',
    )


CHUNK_VOXELS = 2**25
"""Voxels (times volumes) resampled at once, which bounds the memory of NiTransforms."""


def apply_afni_transforms(
    in_file, in_xfm, out_file, max_concurrent=4, clip=False, chunk_voxels=CHUNK_VOXELS
):
    """
    Resample a series with 3dvolreg's ``-1Dmatrix_save`` matrices (NiTransforms).

    The series is read into memory first: NiTransforms reads one volume at a time, and
    each read of a .gz file decompresses it from the start, so the cost grows with the
    square of the number of volumes (538 s against 53 s for 480 volumes).

    It is then resampled a few volumes at a time: NiTransforms computes the target
    coordinates of every volume at once, which took 37 GB on a 280-volume DWI series
    of 146x146x87 voxels. Volumes are resampled independently, so the output is the same.

    With ``clip``, the output is clipped to the range of the input, as 3dvolreg does
    (``-clipit``): cubic splines overshoot at edges, and negative values break the
    logarithm of the entropy focus criterion.

    """
    import nibabel as nb
    from nitransforms.linear import Affine, LinearTransformsMapping, load
    from nitransforms.resampling import apply

    img = nb.load(in_file)
    data = np.asanyarray(img.dataobj)
    if data.ndim == 3:
        data = data[..., np.newaxis]
    xfm = load(in_xfm, fmt='afni', reference=in_file, moving=in_file)
    matrices = np.asanyarray(xfm.matrix).reshape(-1, 4, 4)

    nvols = data.shape[3]
    step = max(1, chunk_voxels // int(np.prod(data.shape[:3])))
    realigned = None
    for start in range(0, nvols, step):
        stop = min(start + step, nvols)
        # NiTransforms takes a single volume as a 3D image, with a single transform
        single = stop - start == 1
        chunk = apply(
            Affine(matrices[start], reference=xfm.reference)
            if single
            else LinearTransformsMapping(matrices[start:stop], reference=xfm.reference),
            img.__class__(
                data[..., start] if single else data[..., start:stop], img.affine, img.header
            ),
            dtype_width=4,
            serialize_nvols=2,
            max_concurrent=max_concurrent,
            mode='reflect',
        )
        values = np.asanyarray(chunk.dataobj)
        if realigned is None:
            # The header keeps the type written to disk, which may be scaled integers
            header = chunk.header
            realigned = np.empty(data.shape, dtype=values.dtype)
        realigned[..., start:stop] = values.reshape(data.shape[:3] + (stop - start,))

    if clip:
        np.clip(realigned, data.min(), data.max(), out=realigned)
    if img.ndim == 3:
        realigned = realigned[..., 0]
    img.__class__(realigned, img.affine, header).to_filename(out_file)
    return str(Path(out_file).absolute())


def _rotation(axis, angles):
    """Right-handed rotations by ``angles`` (radians) about one axis, one per angle."""
    cos, sin = np.cos(angles), np.sin(angles)
    rot = np.zeros((len(angles), 3, 3))
    rot[:, axis, axis] = 1.0
    i, j = (axis + 1) % 3, (axis + 2) % 3
    rot[:, i, i] = cos
    rot[:, j, j] = cos
    rot[:, i, j] = -sin
    rot[:, j, i] = sin
    return rot
