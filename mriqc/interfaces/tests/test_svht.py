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
"""Check the svht_denoise interface, which replaces MRtrix3's dwidenoise."""

import shutil

import nibabel as nb
import numpy as np
import pytest

from mriqc.interfaces.diffusion import SVHTDenoise


def test_svht_cmdline(tmp_path, monkeypatch):
    """The options map onto svht_denoise's dwidenoise-like flags."""
    monkeypatch.chdir(tmp_path)
    for name in ('dwi.nii.gz', 'mask.nii.gz'):
        (tmp_path / name).touch()

    denoise = SVHTDenoise(
        in_file='dwi.nii.gz', mask='mask.nii.gz', noise='noisemap.nii.gz', nthreads=2
    )
    assert denoise.cmdline == (
        'svht_denoise dwi.nii.gz dwi_denoised.nii.gz'
        ' -mask mask.nii.gz -noise noisemap.nii.gz -nthreads 2'
    )
    outputs = denoise._list_outputs()
    assert outputs['out_file'] == str(tmp_path / 'dwi_denoised.nii.gz')
    assert outputs['noise'] == str(tmp_path / 'noisemap.nii.gz')


@pytest.mark.skipif(shutil.which('svht_denoise') is None, reason='svht_denoise not installed')
def test_svht_run(tmp_path, monkeypatch):
    """Denoising keeps the grid, zeroes outside the mask, and maps the noise level."""
    monkeypatch.chdir(tmp_path)
    rng = np.random.default_rng(0)
    signal = np.full((12, 12, 12, 8), 100.0)
    data = np.abs(signal + rng.normal(0, 5, signal.shape)).astype('float32')
    mask = np.zeros(signal.shape[:3], dtype='uint8')
    mask[2:10, 2:10, 2:10] = 1
    nb.Nifti1Image(data, np.eye(4)).to_filename('dwi.nii.gz')
    nb.Nifti1Image(mask, np.eye(4)).to_filename('mask.nii.gz')

    result = SVHTDenoise(
        in_file='dwi.nii.gz', mask='mask.nii.gz', noise='noisemap.nii.gz', nthreads=1
    ).run()

    denoised = np.asanyarray(nb.load(result.outputs.out_file).dataobj)
    noise = np.asanyarray(nb.load(result.outputs.noise).dataobj)
    assert denoised.shape == data.shape
    assert not denoised[mask == 0].any()
    assert denoised[mask > 0].std() < data[mask > 0].std()
    assert noise.shape == mask.shape
    assert noise[mask > 0].mean() > 0
