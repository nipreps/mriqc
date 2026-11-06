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
"""Tests for the selection of input runs."""

import nibabel as nb
import numpy as np


def test_single_volume_anatomical(tmp_path):
    """Anatomical runs with more than one volume are dropped."""
    from importlib import reload

    from mriqc import config
    from mriqc.utils.misc import initialize_meta_and_data

    reload(config)

    bids_dir = tmp_path / 'ds'
    files = []
    for subject, shape in (('01', (5, 5, 5)), ('02', (5, 5, 5, 2))):
        path = bids_dir / f'sub-{subject}' / 'anat' / f'sub-{subject}_T1w.nii.gz'
        path.parent.mkdir(parents=True)
        nb.Nifti1Image(np.zeros(shape, dtype='uint8'), np.eye(4)).to_filename(path)
        files.append(str(path))
    (bids_dir / 'dataset_description.json').write_text(
        '{"Name": "Example dataset", "BIDSVersion": "1.10.0"}'
    )

    config.execution.output_dir = tmp_path / 'out'
    config.execution.work_dir = tmp_path / 'work'
    config.execution.bids_dir = bids_dir
    config.execution.init()
    config.loggers.init()
    config.workflow.inputs = {'t1w': files}

    initialize_meta_and_data()

    assert list(config.workflow.inputs['t1w']) == files[:1]
