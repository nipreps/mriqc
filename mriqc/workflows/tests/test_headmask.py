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
"""Check the head mask computed from the gradient magnitude (``gradient_threshold``)."""

import nibabel as nb
import numpy as np

from mriqc.workflows.anatomical.base import gradient_threshold


def test_head_mask_keeps_the_brain_at_the_fov_edge(tmp_path):
    """A brain cut by the FoV stays in the head mask, and nothing else is added (#1475)."""
    # A head and its brain run through the whole volume along z, as when a reduced FoV
    # cuts the skull at the vertex and below the cerebellum
    shape = (48, 48, 24)
    xx, yy = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
    radius = np.hypot(xx - 24, yy - 24)[..., np.newaxis].repeat(shape[2], axis=2)
    brain = radius < 12
    # As ``image_gradient`` writes it: the scalp's edge, and 100 within the brain mask
    grad = np.zeros(shape, dtype=np.float32)
    grad[(radius > 16) & (radius < 19)] = 100.0
    grad[brain] = 100.0

    affine = np.diag([1.0, 1.0, 3.0, 1.0])
    nb.Nifti1Image(grad, affine).to_filename(tmp_path / 'grad.nii.gz')
    nb.Nifti1Image(brain.astype(np.uint8), affine).to_filename(tmp_path / 'brainmask.nii.gz')

    out_file = gradient_threshold(
        str(tmp_path / 'grad.nii.gz'),
        str(tmp_path / 'brainmask.nii.gz'),
        out_file=str(tmp_path / 'headmask.nii.gz'),
    )
    head = np.asanyarray(nb.load(out_file).dataobj) > 0

    # The brain is in the head mask, also on the first and last slices
    assert not (brain & ~head).any()
    # Air stays out: no planes at the volume's edges, and the corners are background
    assert not head[0].any()
    assert not head[:, 0].any()
    assert not head[0, 0, :].any()
