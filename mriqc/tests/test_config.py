# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# Copyright 2021 The NiPreps Developers <nipreps@gmail.com>
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
"""Exercise config module."""

from pathlib import Path

import pytest


def _expand_bids(tmp_path, testdata_path, testcase):
    """Expand manifest file into a temporal folder."""

    text = (testdata_path / f'{testcase}.manifest').read_text().splitlines()
    root = Path(text[0].strip())
    out_path = tmp_path / testcase

    for path in reversed(text[1:]):
        relpath = Path(path).relative_to(root)
        if '.' in relpath.name or relpath in ('CHANGES', 'README', 'LICENSE'):
            (out_path / relpath.parent).mkdir(parents=True, exist_ok=True)
            if not (out_path / relpath).exists():
                contents = '{}' if relpath.name.endswith('.json') else ''
                (out_path / relpath).write_text(contents)
        else:
            (out_path / relpath).mkdir(parents=True, exist_ok=True)

    if (
        not (out_path / 'dataset_description.json').exists()
        or 'Name' not in (out_path / 'dataset_description.json').read_text()
    ):
        (out_path / 'dataset_description.json').write_text(
            '{"Name": "Example dataset", "BIDSVersion": "1.0.2"}'
        )

    return out_path


@pytest.mark.parametrize(
    'testcase',
    [
        'gh921-dmd-20220428-0',
        'gh921-dmd-20230319-0',
        'gh1086-ds004134',
    ],
)
def test_bids_indexing_manifest(tmp_path, testdata_path, testcase):
    """Check ``BIDSLayout`` is indexing what it should."""

    from importlib import reload

    from mriqc import config

    reload(config)

    config.execution.output_dir = Path(tmp_path) / 'out'
    config.execution.bids_dir = _expand_bids(
        tmp_path,
        testdata_path,
        testcase,
    )
    config.execution.init()
    assert len(config.execution.layout.get()) == int(
        (testdata_path / f'{testcase}.oracle').read_text().strip()
    )


@pytest.mark.parametrize('participant_label', [None, ['01']])
def test_bids_inheritance(tmp_path, participant_label):
    """Check that metadata in top-level sidecars reach the input files."""
    from importlib import reload

    from mriqc import config

    reload(config)

    from niworkflows.utils.bids import DEFAULT_BIDS_QUERIES, collect_data

    bids_dir = tmp_path / 'ds'
    for path in (
        'sub-01/anat/sub-01_FLAIR.nii.gz',
        'sub-01/dwi/sub-01_dwi.nii.gz',
        'sub-01/func/sub-01_task-rest_bold.nii.gz',
    ):
        (bids_dir / path).parent.mkdir(parents=True, exist_ok=True)
        (bids_dir / path).write_text('')
    (bids_dir / 'dataset_description.json').write_text(
        '{"Name": "Example dataset", "BIDSVersion": "1.10.0"}'
    )
    (bids_dir / 'FLAIR.json').write_text('{"InversionTime": 2.5}')
    (bids_dir / 'dwi.json').write_text('{"PhaseEncodingDirection": "j-"}')
    (bids_dir / 'task-rest_bold.json').write_text('{"RepetitionTime": 2.0}')
    (bids_dir / 'sub-01' / 'dwi' / 'sub-01_dwi.json').write_text('{"EchoTime": 0.09}')

    config.execution.output_dir = tmp_path / 'out'
    config.execution.work_dir = tmp_path / 'work'
    config.execution.bids_dir = bids_dir
    config.execution.participant_label = participant_label
    config.execution.init()

    layout = config.execution.layout
    assert layout.get_metadata(bids_dir / 'sub-01' / 'dwi' / 'sub-01_dwi.nii.gz') == {
        'EchoTime': 0.09,
        'PhaseEncodingDirection': 'j-',
    }
    assert layout.get_metadata(bids_dir / 'sub-01' / 'func' / 'sub-01_task-rest_bold.nii.gz') == {
        'RepetitionTime': 2.0
    }

    # FLAIR files are indexed, inherit the top-level sidecar, and are selected as inputs
    flair = bids_dir / 'sub-01' / 'anat' / 'sub-01_FLAIR.nii.gz'
    assert layout.get_metadata(flair) == {'InversionTime': 2.5}
    inputs, _ = collect_data(layout, ['01'], queries={'flair': DEFAULT_BIDS_QUERIES['flair']})
    assert inputs['flair'] == [str(flair)]


def _init_layout(bids_dir, work_dir, **kwargs):
    """Index ``bids_dir`` with a fresh config and return the config module."""
    from importlib import reload

    from mriqc import config

    reload(config)
    config.execution.output_dir = work_dir.parent / 'out'
    config.execution.work_dir = work_dir
    config.execution.bids_dir = bids_dir
    for key, value in kwargs.items():
        setattr(config.execution, key, value)
    config.execution.init()
    return config


def _make_dataset(bids_dir):
    (bids_dir / 'sub-01' / 'anat').mkdir(parents=True)
    (bids_dir / 'sub-01' / 'anat' / 'sub-01_T1w.nii.gz').write_text('')
    (bids_dir / 'dataset_description.json').write_text(
        '{"Name": "Example dataset", "BIDSVersion": "1.10.0"}'
    )
    return bids_dir


@pytest.mark.parametrize('relocate', ['move', 'copy'])
def test_bids_database_relocated(tmp_path, relocate):
    """A default database indexing another path is rebuilt (gh-1410)."""
    import shutil

    old_dir = _make_dataset(tmp_path / 'ds')
    config = _init_layout(old_dir, tmp_path / 'work')
    assert config.execution.bids_database_dir == tmp_path / 'work' / '.bids_db'
    assert not (tmp_path / 'out' / '.bids_db').exists()

    new_dir = tmp_path / 'moved'
    (shutil.move if relocate == 'move' else shutil.copytree)(old_dir, new_dir)
    config = _init_layout(new_dir, tmp_path / 'work')
    assert config.execution.layout.root == str(new_dir)
    assert config.execution.layout.get(suffix='T1w', return_type='file') == [
        str(new_dir / 'sub-01' / 'anat' / 'sub-01_T1w.nii.gz')
    ]


def test_bids_database_user_mismatch(tmp_path):
    """A database given with ``--bids-database-dir`` is never re-indexed."""
    user_db = tmp_path / 'user_db'
    _init_layout(_make_dataset(tmp_path / 'ds'), tmp_path / 'work', bids_database_dir=user_db)

    with pytest.raises(ValueError, match='indexes a different dataset'):
        _init_layout(
            _make_dataset(tmp_path / 'other'), tmp_path / 'work', bids_database_dir=user_db
        )
    assert (user_db / 'layout_index.sqlite').exists()


def test_bids_database_wipe(tmp_path):
    """``--bids-database-wipe`` replaces an existing default database."""
    bids_dir = _make_dataset(tmp_path / 'ds')
    _init_layout(bids_dir, tmp_path / 'work')
    config = _init_layout(bids_dir, tmp_path / 'work', bids_database_wipe=True)
    assert config.execution.layout.root == str(bids_dir)
    assert sorted(p.name for p in (tmp_path / 'work').iterdir()) == ['.bids_db']


def test_seeds(monkeypatch):
    """A master seed fixes the ANTs and NumPy seeds, and nothing is seeded without one."""
    import os

    from mriqc import config

    # Restore the environment and the seeds after the test
    monkeypatch.setenv('ANTS_RANDOM_SEED', 'unset')
    monkeypatch.delenv('ANTS_RANDOM_SEED')
    for name in ('_random_seed', 'master', 'ants', 'numpy'):
        monkeypatch.setattr(config.seeds, name, None)

    config.seeds.init()
    assert (config.seeds.master, config.seeds.ants, config.seeds.numpy) == (None, None, None)
    assert 'ANTS_RANDOM_SEED' not in os.environ

    config.seeds.load({'_random_seed': 42})
    seeds = (config.seeds.ants, config.seeds.numpy)
    assert config.seeds.master == 42
    assert None not in seeds
    assert os.environ['ANTS_RANDOM_SEED'] == str(config.seeds.ants)

    # Worker processes load the seeds from the config file and get the same ones
    saved = config.get()['seeds']
    assert saved == {'master': 42, 'ants': seeds[0], 'numpy': seeds[1]}
    for name in ('_random_seed', 'master', 'ants', 'numpy'):
        setattr(config.seeds, name, None)
    config.seeds.load(saved)
    assert (config.seeds.ants, config.seeds.numpy) == seeds
