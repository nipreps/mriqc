# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# Copyright 2024 The NiPreps Developers <nipreps@gmail.com>
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

import sys

import pytest

from mriqc.__main__ import main


@pytest.fixture(autouse=True)
def set_command(monkeypatch):
    with monkeypatch.context() as m:
        m.setattr(sys, 'argv', ['mriqc'])
        yield


def test_help(capsys):
    with pytest.raises(SystemExit):
        main(['--help'])
    captured = capsys.readouterr()
    # The program name depends on how Python was started (``python -m pytest`` on 3.14+)
    assert captured.out.startswith('usage: ')
    assert '[-h]' in captured.out.splitlines()[0]


def test_main(tmp_path):
    bids_dir = tmp_path / 'data/sub-01'
    out_path = tmp_path / 'out'

    with pytest.raises(SystemExit):
        main([str(bids_dir), str(out_path)])

    analysis_level = 'participant'
    species = 'human'

    with pytest.raises(SystemExit):
        main(
            [
                str(bids_dir),
                str(out_path),
                analysis_level,
                '--species',
                species,
            ]
        )


@pytest.fixture
def dropped_bids(tmp_path):
    """A BIDS dataset whose image content is absent, as after ``datalad drop``."""
    bids_dir = tmp_path / 'ds'
    (bids_dir / 'sub-01' / 'anat').mkdir(parents=True)
    (bids_dir / 'sub-01' / 'anat' / 'sub-01_T1w.nii.gz').symlink_to(
        '../../.git/annex/objects/sub-01_T1w.nii.gz'
    )
    (bids_dir / 'dataset_description.json').write_text(
        '{"Name": "Example dataset", "BIDSVersion": "1.10.0"}'
    )
    return bids_dir


def _no_raw_data():
    raise AssertionError('The group level must not read the input images.')


def test_group_level_without_raw_data(tmp_path, monkeypatch, cli_config, dropped_bids):
    """The group level does not read the input images (#1388)."""
    from mriqc.cli.parser import parse_args

    monkeypatch.setattr('mriqc.utils.misc.initialize_meta_and_data', _no_raw_data)
    parse_args([str(dropped_bids), str(tmp_path / 'out'), 'group', '-w', str(tmp_path / 'work')])

    assert cli_config.workflow.analysis_level == ['group']
    assert cli_config.workflow.inputs == {
        't1w': [str(dropped_bids / 'sub-01' / 'anat' / 'sub-01_T1w.nii.gz')]
    }
    # The config is written out without the inputs' metadata
    cli_config.to_filename()


def test_group_level_no_iqms(tmp_path, monkeypatch, cli_config, dropped_bids):
    """Without IQM files, the group level says where it looked and for what."""
    monkeypatch.setattr('mriqc.utils.misc.initialize_meta_and_data', _no_raw_data)
    out_dir = tmp_path / 'out'
    with pytest.raises(Exception, match=r'\+iqms\.parquet') as excinfo:
        main([str(dropped_bids), str(out_dir), 'group', '-w', str(tmp_path / 'work')])

    assert f'<{out_dir}>' in str(excinfo.value)
