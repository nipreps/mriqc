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

import pytest
from pathlib import Path

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
