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
"""Tests for the dfcheck command."""

from pathlib import Path

import pandas as pd
import pytest

REFERENCE = Path(__file__).parent.parent / 'data' / 'testdata' / 'group_dwi.tsv'


@pytest.mark.parametrize(
    ('change', 'strict', 'fails'),
    [
        (1.0, True, False),
        (1.1, False, False),  # two images always correlate perfectly
        (1.1, True, True),
    ],
)
@pytest.mark.filterwarnings('ignore::RuntimeWarning')
def test_dfcheck_strict(tmp_path, monkeypatch, change, strict, fails):
    """``--strict`` fails on any changed value, not only on poorly correlated IQMs."""
    from mriqc.bin import dfcheck

    table = pd.read_csv(REFERENCE, sep='\t')
    table.loc[0, 'snr_cc_shell2_best'] *= change
    table.to_csv(tmp_path / 'group.tsv', sep='\t', index=False)

    argv = ['dfcheck', '-i', str(tmp_path / 'group.tsv'), '-r', str(REFERENCE)]
    monkeypatch.setattr('sys.argv', argv + (['--strict'] if strict else []))
    with pytest.raises(SystemExit) as exit_info:
        dfcheck.main()

    assert (exit_info.value.code not in (0, None)) is fails
