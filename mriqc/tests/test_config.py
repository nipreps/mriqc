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
