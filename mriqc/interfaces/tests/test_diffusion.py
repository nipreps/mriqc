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
"""Tests for the DWI interfaces."""

from pathlib import Path

import pytest

from mriqc.interfaces.diffusion import DWISummary

B_DICT = {0: [0, 9], 1000: [1, 2, 3, 4], 2000: [5, 6, 7, 8]}


def _summary(tmp_path, monkeypatch, **inputs):
    monkeypatch.chdir(tmp_path)
    in_file = tmp_path / 'sub-01_dwi.nii.gz'
    in_file.write_bytes(b'')
    inputs = {'n_shells': 2, 'is_dsi': False, 'b_dict': B_DICT} | inputs
    result = DWISummary(in_file=str(in_file), **inputs).run()
    return Path(result.outputs.out_report).read_text()


def test_dwi_summary(tmp_path, monkeypatch):
    """The summary renders the metadata and shells it is given."""
    html = _summary(tmp_path, monkeypatch, pe_direction='j-', echo_time=0.089)

    assert '<li>Filename: sub-01_dwi.nii.gz</li>' in html
    assert '<li>Phase-encoding (PE) direction: j-</li>' in html
    assert '<li>Echo time (TE): 89 ms</li>' in html
    assert '<li>Repetition time (TR): n/a</li>' in html
    assert '<li>Number of b=0: 2</li>' in html
    assert '<li>Indices of the b=0 scans: 0, 9</li>' in html
    assert '<li>Type of DWIs: multi-shell</li>' in html
    assert '<li>Number of DWIs: 4 at b=1000, 4 at b=2000</li>' in html


@pytest.mark.parametrize(
    ('n_shells', 'is_dsi', 'dwi_type'),
    [(1, False, 'single-shell'), (2, False, 'multi-shell'), (3, True, 'DSI')],
)
def test_dwi_summary_type(tmp_path, monkeypatch, n_shells, is_dsi, dwi_type):
    """The type of DWI follows the shells found upstream."""
    html = _summary(tmp_path, monkeypatch, n_shells=n_shells, is_dsi=is_dsi)
    assert f'<li>Type of DWIs: {dwi_type}</li>' in html
