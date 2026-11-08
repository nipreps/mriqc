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

"""Test parser."""

from pathlib import Path

import pytest

from mriqc.cli.parser import _build_parser

MIN_ARGS = ['bids_dir', 'out', 'participant']


def _create_set_bids_dir(args, _tmp_path):
    datapath = _tmp_path / MIN_ARGS[0]
    datapath.mkdir(exist_ok=True)
    args[0] = str(datapath)
    return args


@pytest.mark.parametrize(
    ('args', 'code'),
    [
        ([], 2),
    ],
)
def test_parser_errors(args, code):
    """Check behavior of the parser."""
    with pytest.raises(SystemExit) as error:
        _build_parser().parse_args(args)

    assert error.value.code == code


@pytest.mark.parametrize(
    'args',
    [
        MIN_ARGS,
    ],
)
def test_parser_valid(tmp_path, args):
    """Check valid arguments."""
    args = _create_set_bids_dir(args, tmp_path)

    opts = _build_parser().parse_args(args)

    assert opts.bids_dir == Path(args[0])


@pytest.mark.parametrize(
    ('argdest', 'argstr', 'argval'),
    [
        ('verbose_count', '-v', 1),
        ('verbose_count', '-vv', 2),
        ('verbose_count', '-vvv', 3),
    ],
)
def test_verbosity_arg(tmp_path, argdest, argstr, argval):
    """Check the correct parsing of the verbosity argument."""
    args = MIN_ARGS + [argstr]

    args = _create_set_bids_dir(args, tmp_path)

    opts = _build_parser().parse_args(args)

    assert getattr(opts, argdest) == argval


@pytest.mark.parametrize(
    ('argval', '_species'),
    [
        ('human', 'human'),
        ('rat', 'rat'),
    ],
)
def test_species_arg(tmp_path, argval, _species):
    """Check the correct parsing of the species argument."""
    args = MIN_ARGS + ['--species', argval]

    args = _create_set_bids_dir(args, tmp_path)

    opts = _build_parser().parse_args(args)

    assert opts.species == _species


def test_random_seed_arg(tmp_path):
    """Check the correct parsing of the random seed argument."""
    args = _create_set_bids_dir(MIN_ARGS + ['--random-seed', '42'], tmp_path)
    assert _build_parser().parse_args(args)._random_seed == 42

    args = _create_set_bids_dir(list(MIN_ARGS), tmp_path)
    assert _build_parser().parse_args(args)._random_seed is None


@pytest.mark.parametrize(
    ('bids_filters', 'part'),
    [
        (None, 'mag'),
        ({'dwi': {'part': 'phase'}}, 'phase'),
    ],
)
def test_dwi_phase(tmp_path, monkeypatch, cli_config, bids_filters, part):
    """DWI phase images are only processed when the BIDS filters select them (#1436)."""
    import json
    from copy import deepcopy

    from niworkflows.utils.bids import DEFAULT_BIDS_QUERIES

    from mriqc.cli.parser import parse_args

    bids_dir = tmp_path / 'ds'
    dwi_dir = bids_dir / 'sub-01' / 'dwi'
    dwi_dir.mkdir(parents=True)
    for name in ('part-mag_dwi', 'part-phase_dwi'):
        for ext in ('nii.gz', 'bval', 'bvec'):
            (dwi_dir / f'sub-01_{name}.{ext}').write_text('')
    (bids_dir / 'dataset_description.json').write_text(
        '{"Name": "Example dataset", "BIDSVersion": "1.10.0"}'
    )

    args = [str(bids_dir), str(tmp_path / 'out'), 'participant', '-m', 'dwi']
    args += ['-w', str(tmp_path / 'work')]
    if bids_filters:
        (tmp_path / 'filters.json').write_text(json.dumps(bids_filters))
        args += ['--bids-filter-file', str(tmp_path / 'filters.json')]

    # These empty files have no data or metadata to read
    monkeypatch.setattr('mriqc.utils.misc.initialize_meta_and_data', lambda: None)
    defaults = deepcopy(DEFAULT_BIDS_QUERIES)
    parse_args(args)

    assert cli_config.workflow.inputs == {'dwi': [str(dwi_dir / f'sub-01_part-{part}_dwi.nii.gz')]}
    # The BIDS filters do not leak into niworkflows' defaults
    assert DEFAULT_BIDS_QUERIES == defaults


def test_task_id(tmp_path, monkeypatch, cli_config):
    """``--task-id`` selects BOLD runs (it raised ``KeyError: 'pet'``)."""
    from mriqc.cli.parser import parse_args

    bids_dir = tmp_path / 'ds'
    func_dir = bids_dir / 'sub-01' / 'func'
    func_dir.mkdir(parents=True)
    for task in ('rest', 'nback'):
        (func_dir / f'sub-01_task-{task}_bold.nii.gz').write_text('')
    (bids_dir / 'dataset_description.json').write_text(
        '{"Name": "Example dataset", "BIDSVersion": "1.10.0"}'
    )

    # These empty files have no data or metadata to read
    monkeypatch.setattr('mriqc.utils.misc.initialize_meta_and_data', lambda: None)
    parse_args(
        [str(bids_dir), str(tmp_path / 'out'), 'participant', '-m', 'bold', '--task-id', 'rest']
        + ['-w', str(tmp_path / 'work')]
    )

    assert cli_config.workflow.inputs == {'bold': [str(func_dir / 'sub-01_task-rest_bold.nii.gz')]}
