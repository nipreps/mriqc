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
import pytest


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


def test_generate_tsv_stems(tmp_path):
    """The group table only gathers the IQM files with the given stems (#812, #1353)."""
    import pandas as pd

    from mriqc.utils.misc import generate_tsv

    names = ['sub-01_task-rest_bold', 'sub-01_task-stroop_bold', 'sub-02_task-rest_bold']
    for name in names:
        path = tmp_path / name[:6] / 'func' / f'{name}+iqms.parquet'
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({'fd_mean': [0.1]}).to_parquet(path, index=False)

    dataframe, _ = generate_tsv(tmp_path, 'bold')
    assert dataframe['bids_name'].tolist() == names

    dataframe, out_tsv = generate_tsv(tmp_path, 'bold', stems={f'{names[2]}+iqms'})
    assert out_tsv == tmp_path / 'group_bold.tsv'
    assert pd.read_csv(out_tsv, sep='\t')['bids_name'].tolist() == names[2:]


@pytest.fixture
def echos(tmp_path):
    """Set the config as the parser leaves it for one multi-echo run of two subjects."""
    from importlib import reload

    from mriqc import config

    reload(config)

    for subject in ('01', '02'):
        (tmp_path / 'ds' / f'sub-{subject}').mkdir(parents=True)
    func_dir = tmp_path / 'ds' / 'sub-01' / 'func'
    files = [str(func_dir / f'sub-01_task-rest_echo-{echo}_bold.nii.gz') for echo in (1, 2)]
    config.execution.bids_dir = tmp_path / 'ds'
    config.execution.bids_filters = {}
    # Without --participant-label, the parser selects all the subjects
    config.execution.participant_label = ['01', '02']
    config.workflow.inputs = {'bold': [files]}
    return files


def test_unselected_iqm_stems(echos):
    """Without selection, the group level gathers all the IQM files."""
    from mriqc.cli.run import _selected_iqm_stems

    assert _selected_iqm_stems() is None


@pytest.mark.parametrize(
    ('option', 'value'),
    [
        ('participant_label', ['01']),
        ('session_id', ['1']),
        ('run_id', [1]),
        ('task_id', ['rest']),
        ('bids_filters', {'bold': {'echo': 1}}),
    ],
)
def test_selected_iqm_stems(tmp_path, echos, option, value):
    """With a selection, the group level gathers the IQM files of the inputs."""
    from pathlib import Path

    from mriqc import config
    from mriqc.cli.run import _selected_iqm_stems
    from mriqc.interfaces.bids import IQMFileSink

    setattr(config.execution, option, value)
    stems = _selected_iqm_stems()
    assert stems == {
        'sub-01_task-rest_echo-1_bold+iqms',
        'sub-01_task-rest_echo-2_bold+iqms',
    }
    # Each stem names the IQM file of an input
    for echo in echos:
        sink = IQMFileSink(in_file=echo, modality='bold', out_dir=str(tmp_path / 'out'))
        assert Path(sink._gen_outfiles()[0]).stem in stems
