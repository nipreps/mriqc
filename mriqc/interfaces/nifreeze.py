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
"""Head-motion correction of DWI with `NiFreeze <https://github.com/nipreps/nifreeze>`__."""

import tempfile
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import nibabel as nb
import numpy as np
from nipype.interfaces.base import (
    BaseInterfaceInputSpec,
    File,
    SimpleInterface,
    TraitedSpec,
    isdefined,
    traits,
)
from nipype.utils.filemanip import split_filename

from mriqc.utils.hmc import apply_afni_transforms, ras_to_afni, write_afni_motion

B0_THRESHOLD = 50
"""Volumes up to this b-value are b=0 for NiFreeze (``DEFAULT_LOWB_THRESHOLD``)."""


class _NiFreezeHMCInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True, desc='DWI series to realign')
    basefile = File(
        exists=True,
        mandatory=True,
        desc='b=0 reference on the grid of in_file, to which the b=0 volumes are registered',
    )
    in_bvec = File(exists=True, desc='b-vectors (BIDS)')
    in_bval = File(exists=True, desc='b-values (BIDS); without them, every volume is a b=0')
    brain_mask = File(exists=True, desc='brain mask on the grid of in_file')
    seed = traits.Int(desc='random seed of the volume order and of ANTs')
    num_threads = traits.Int(1, usedefault=True, nohash=True, desc='threads of ANTs')
    estimate = traits.Bool(
        False,
        usedefault=True,
        desc='estimate the motion only, without writing the realigned series',
    )
    eddy = traits.Bool(
        False,
        usedefault=True,
        desc='also correct eddy-current distortions, with a second, affine registration of '
        'each diffusion-weighted volume onto its prediction',
    )


class _NiFreezeHMCOutputSpec(TraitedSpec):
    out_file = File(desc='realigned series (unless estimate is set)')
    oned_file = File(exists=True, desc="motion parameters, in 3dvolreg's -1Dfile format")
    oned_matrix_save = File(exists=True, desc="matrices, in 3dvolreg's -1Dmatrix_save format")


class NiFreezeHMC(SimpleInterface):
    """
    Estimate head motion of a DWI series with NiFreeze, writing 3dvolreg's outputs.

    Each diffusion-weighted volume is registered (ANTs) onto NiFreeze's prediction of
    that same volume from the others (the GQI-KRR model), instead of onto the b=0
    reference. The b=0 volumes are registered onto the reference, with NiFreeze's
    b0-to-b0 settings; a single b=0 is the reference, so it does not move.

    The outputs match those of nipype's ``Volreg`` that MRIQC uses: the motion
    parameters in 3dvolreg's ``-1Dfile`` format, its ``-1Dmatrix_save`` matrices,
    and the series resampled with them.

    With ``eddy``, each diffusion-weighted volume is registered onto its prediction
    a second time, starting from the rigid estimate, with an affine transform
    (NiFreeze's ``dwi-to-dwi_level1`` settings) that also takes up the scaling, shear
    and shifts of eddy currents. The matrices and the realigned series are then those
    of the affine pass, and the motion parameters (hence the framewise displacement)
    those of the rigid one.

    """

    input_spec = _NiFreezeHMCInputSpec
    output_spec = _NiFreezeHMCOutputSpec

    def _run_interface(self, runtime):
        img = nb.load(self.inputs.in_file)
        nvols = 1 if img.ndim < 4 else img.shape[3]
        bvals = (
            np.loadtxt(self.inputs.in_bval, ndmin=1)
            if isdefined(self.inputs.in_bval)
            else np.zeros(nvols)
        )
        if bvals.size != nvols:
            raise ValueError(f'{bvals.size} b-values for {nvols} volumes')
        lowb = bvals <= B0_THRESHOLD
        if not lowb.all() and not isdefined(self.inputs.in_bvec):
            raise ValueError('Diffusion-weighted volumes need their b-vectors (in_bvec)')

        seed = self.inputs.seed if isdefined(self.inputs.seed) else None
        # Rigid estimates (the motion parameters), and the transforms that realign
        # the series (affine with eddy-current correction)
        motion = np.tile(np.eye(4), (nvols, 1, 1))
        transforms = motion.copy()

        # NiFreeze works in temporary directories, and prints progress bars
        tempdir = tempfile.tempdir
        tempfile.tempdir = runtime.cwd
        try:
            with (
                open(Path(runtime.cwd) / 'nifreeze.log', 'w') as log,
                redirect_stdout(log),
                redirect_stderr(log),
            ):
                if not lowb.all():
                    motion[~lowb], transforms[~lowb] = self._estimate_dwi(seed)
                if lowb.sum() > 1:
                    # Without diffusion gradients, b=0 volumes have no eddy currents
                    motion[lowb] = transforms[lowb] = self._register_b0s(
                        img, np.flatnonzero(lowb), seed, runtime
                    )
        finally:
            tempfile.tempdir = tempdir

        _, stem, _ = split_filename(self.inputs.in_file)
        self._results['oned_file'] = str(Path(runtime.cwd) / f'{stem}.1D')
        self._results['oned_matrix_save'] = str(Path(runtime.cwd) / f'{stem}.aff12.1D')
        write_afni_motion(
            ras_to_afni(transforms, img.header),
            img.header,
            img.shape[:3],
            self._results['oned_file'],
            self._results['oned_matrix_save'],
            tool='NiFreeze (affine, eddy currents)' if self.inputs.eddy else 'NiFreeze',
            rigid=ras_to_afni(motion, img.header) if self.inputs.eddy else None,
        )

        if not self.inputs.estimate:
            out_file = Path(runtime.cwd) / f'{stem}_hmc.nii.gz'
            if np.allclose(transforms, np.eye(4)):
                img.to_filename(out_file)  # nothing moved (e.g., a single b=0)
            else:
                apply_afni_transforms(
                    self.inputs.in_file,
                    self._results['oned_matrix_save'],
                    out_file,
                    max_concurrent=self.inputs.num_threads,
                    clip=True,  # as 3dvolreg does
                )
            self._results['out_file'] = str(out_file)
        return runtime

    def _estimate_dwi(self, seed):
        """
        Register each diffusion-weighted volume onto its GQI-KRR prediction.

        Returns the rigid matrices, and those that realign the volumes: the same, or
        with ``eddy`` the affine ones (RAS+, reference to moving).

        """
        from nifreeze.data.dmri import from_nii
        from nifreeze.estimator import Estimator

        dwi = from_nii(
            self.inputs.in_file,
            brainmask_file=self.inputs.brain_mask if isdefined(self.inputs.brain_mask) else None,
            bvec_file=self.inputs.in_bvec,
            bval_file=self.inputs.in_bval,
            b0_file=self.inputs.basefile,
        )
        run = {
            'omp_nthreads': self.inputs.num_threads,
            'n_jobs': self.inputs.num_threads,
            'seed': seed,
        }
        Estimator('gqi-krr').run(dwi, **run)
        rigid = np.array(dwi.motion_affines)  # a copy, kept as the affine pass updates them
        if not self.inputs.eddy:
            return rigid, rigid

        # The affine pass starts from the rigid estimates, which it reads from the dataset
        Estimator('gqi-krr', ants_config='dwi-to-dwi_level1').run(dwi, **run)
        return rigid, np.array(dwi.motion_affines)

    def _register_b0s(self, img, indices, seed, runtime):
        """Register each b=0 volume onto the reference (RAS+, reference to moving)."""
        from nifreeze.registration.ants import generate_command
        from nitransforms.io.itk import ITKLinearTransform

        fixed = self.inputs.basefile
        matrices = []
        for i in indices:
            moving = Path(runtime.cwd) / f'b0_{i:04d}.nii'
            volume = np.asanyarray(img.dataobj[..., i] if img.ndim > 3 else img.dataobj)
            nb.Nifti1Image(volume.astype('float32'), img.affine).to_filename(moving)
            registration = generate_command(
                fixed,
                moving,
                fixedmask_path=(
                    self.inputs.brain_mask if isdefined(self.inputs.brain_mask) else None
                ),
                default='b0-to-b0_level0',
                terminal_output='file_split',
                num_threads=self.inputs.num_threads,
                environ={} if seed is None else {'ANTS_RANDOM_SEED': str(seed)},
                output_transform_prefix=f'b0-{i:04d}_',
            )
            result = registration.run(cwd=runtime.cwd).outputs
            matrices.append(
                ITKLinearTransform.from_filename(result.forward_transforms[0]).to_ras(
                    reference=fixed, moving=str(moving)
                )
            )
        return np.stack(matrices)
