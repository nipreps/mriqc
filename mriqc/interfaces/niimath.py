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
"""Interfaces to `niimath <https://github.com/rordenlab/niimath>`__."""

from itertools import permutations
from pathlib import Path

import nibabel as nb
import numpy as np
from nipype.interfaces.base import (
    CommandLine,
    CommandLineInputSpec,
    File,
    TraitedSpec,
    traits,
)


class _MocoInputSpec(CommandLineInputSpec):
    in_file = File(
        exists=True,
        mandatory=True,
        argstr='%s -moco',
        position=0,
        copyfile=False,
        desc='4D series to realign',
    )
    basefile = File(
        exists=True,
        argstr='-ref %s',
        position=1,
        desc='reference image on the grid of in_file (default: the first volume of in_file)',
    )
    oned_file = File(
        argstr='-1Dfile %s',
        position=2,
        name_source='in_file',
        name_template='%s.1D',
        keep_extension=False,
        desc="motion parameters, in 3dvolreg's -1Dfile format",
    )
    out_file = File(
        argstr='%s',
        position=3,
        name_source='in_file',
        name_template='%s_moco.nii.gz',
        keep_extension=False,
        desc='realigned series',
    )
    estimate = traits.Bool(
        False,
        usedefault=True,
        desc='estimate the motion parameters only, without writing the realigned series',
    )
    num_threads = traits.Int(1, usedefault=True, nohash=True, desc='number of OpenMP threads')


class _MocoOutputSpec(TraitedSpec):
    out_file = File(desc='realigned series (unless estimate is set)')
    oned_file = File(exists=True, desc='motion parameters (roll pitch yaw dS dL dP)')
    oned_matrix_save = File(
        exists=True, desc="3dvolreg's -1Dmatrix_save matrices, from the motion parameters"
    )


class Moco(CommandLine):
    """
    Realign a 4D series with niimath's ``-moco``, a reimplementation of AFNI's ``3dvolreg``.

    Its outputs match those of nipype's ``Volreg`` that MRIQC uses: the motion parameters
    in 3dvolreg's ``-1Dfile`` format, and the matrices 3dvolreg's ``-1Dmatrix_save`` would
    write, computed from those parameters (niimath does not write them).

    """

    _cmd = 'niimath'
    input_spec = _MocoInputSpec
    output_spec = _MocoOutputSpec
    _two_volumes = None

    def _format_arg(self, name, spec, value):
        if name == 'in_file' and self._two_volumes:
            return spec.argstr % self._two_volumes
        if name == 'oned_file' and self.inputs.estimate:
            # With -estimate, the parameter file is the output name
            return f'-estimate {value}'
        return super()._format_arg(name, spec, value)

    def _parse_inputs(self, skip=None):
        if self.inputs.estimate:
            skip = [*(skip or []), 'out_file']
        return super()._parse_inputs(skip=skip)

    def _run_interface(self, runtime):
        self.inputs.environ['OMP_NUM_THREADS'] = f'{self.inputs.num_threads}'
        # niimath picks the output format from FSLOUTPUTTYPE, whatever the extension
        self.inputs.environ['FSLOUTPUTTYPE'] = 'NIFTI_GZ'

        img = nb.load(self.inputs.in_file)
        single = img.ndim < 4 or img.shape[3] == 1
        if single:
            # -moco needs two volumes or more (3dvolreg takes one): realign the volume
            # repeated twice, and keep the first. Each volume is fitted on its own, onto
            # the reference, or onto itself (no motion) without one.
            data = np.asanyarray(img.dataobj).reshape(img.shape[:3] + (1,))
            self._two_volumes = str(Path(runtime.cwd) / 'moco_two_volumes.nii.gz')
            img.__class__(np.repeat(data, 2, axis=3), img.affine, img.header).to_filename(
                self._two_volumes
            )

        runtime = super()._run_interface(runtime)
        if runtime.returncode != 0:
            # Report niimath's error, not the missing outputs it leaves
            self.raise_exception(runtime)

        outputs = self._list_outputs()
        oned_file = Path(outputs['oned_file'])
        if single:
            oned_file.write_text(oned_file.read_text().splitlines(keepends=True)[0])
            if not self.inputs.estimate:
                out = nb.load(outputs['out_file'])
                first = np.asanyarray(out.dataobj)[..., 0].reshape(img.shape)
                out.__class__(first, out.affine, out.header).to_filename(outputs['out_file'])

        matrices = afni_matrices(np.loadtxt(oned_file, ndmin=2), img.header, img.shape[:3])
        np.savetxt(
            outputs['oned_matrix_save'],
            matrices.reshape(-1, 12),
            fmt='%.8g',
            header='3dvolreg matrices (DICOM-to-DICOM, row-by-row), from niimath -moco',
        )
        return runtime

    def _list_outputs(self):
        outputs = super()._list_outputs()
        if self.inputs.estimate:
            outputs.pop('out_file', None)
        oned_file = Path(outputs['oned_file'])
        outputs['oned_matrix_save'] = str(oned_file.with_name(f'{oned_file.stem}.aff12.1D'))
        return outputs


def afni_matrices(params, header, shape):
    """
    Compute the matrices of 3dvolreg's ``-1Dmatrix_save`` from its motion parameters.

    3dvolreg and niimath ignore obliquity: they snap each voxel axis onto the nearest
    DICOM axis (+x Left, +y Posterior, +z Superior), and rotate about the centre of that
    grid. Each matrix maps a point of the base (DICOM mm) onto the input volume.

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
    # The sform, unless the qform has a higher code (as niimath reads them)
    sform, sform_code = header.get_sform(coded=True)
    qform, qform_code = header.get_qform(coded=True)
    affine = sform if sform_code > 0 and not sform_code < qform_code else qform
    zooms = np.abs(np.array(header.get_zooms()[:3], dtype=float))
    zooms[zooms <= 0] = 1.0

    idx2mm = np.diag(zooms)
    origin = np.zeros(3)
    if affine is not None:
        flip = np.array([-1.0, -1.0, 1.0])  # RAS+ to DICOM
        linear = flip[:, None] * affine[:3, :3]
        best = max(
            permutations(range(3)),
            key=lambda perm: sum(abs(linear[perm[j], j]) for j in range(3)),
        )
        idx2mm = np.zeros((3, 3))
        for j, i in enumerate(best):
            idx2mm[i, j] = np.copysign(zooms[j], linear[i, j])
        origin = flip * affine[:3, 3]
    center = origin + idx2mm @ ((np.array(shape[:3]) - 1) / 2)

    params = np.asanyarray(params, dtype=float)
    roll, pitch, yaw = np.deg2rad(params[:, :3]).T
    rotations = _rotation(2, -roll) @ _rotation(0, -pitch) @ _rotation(1, -yaw)
    shifts = params[:, [4, 5, 3]]  # (dL, dP, dS) = DICOM (x, y, z)
    translations = -np.einsum('nij,nj->ni', rotations, shifts) + center - rotations @ center
    return np.concatenate((rotations, translations[..., None]), axis=-1)


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
