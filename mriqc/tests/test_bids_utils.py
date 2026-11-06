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
"""Check the ``.bidsignore`` written into the output folder."""

from fnmatch import fnmatchcase
from pathlib import PurePosixPath

import pytest

from mriqc.utils.bids import write_bidsignore

# Files and folders MRIQC writes that BIDS does not define (see .circleci/circle_*.txt)
OUTPUTS = (
    'sub-01_T1w.html',
    'group_bold.html',
    'group_T1w.tsv',
    'group_variant-failed_bold.csv',
    'logs/mriqc-20260101-120000_0123abcd.log',
    'sub-01/figures',
    'sub-01/figures/sub-01_desc-zoomed_T1w.svg',
    'sub-01/anat/sub-01_T1w+iqms.json',
    'sub-01/ses-1/anat/sub-01_ses-1_FLAIR+iqms.parquet',
    'sub-01/dwi/sub-01_dwi+iqms.json',
    'sub-01/func/sub-01_task-rest_run-1_bold+iqms.parquet',
    'sub-01/func/sub-01_task-rest_run-1_timeseries.tsv',
    'sub-01/func/sub-01_task-rest_run-1_timeseries.json',
)


def _ignored(path, patterns):
    """Match as ``.gitignore`` does for these patterns: by name, at any depth."""
    parts = PurePosixPath(path).parts
    for pattern in patterns:
        if pattern.endswith('/'):
            # Only folders, and so never the last part of a path
            if pattern[:-1] in parts[:-1]:
                return True
        elif any(fnmatchcase(part, pattern) for part in parts):
            return True
    return False


@pytest.fixture
def patterns(tmp_path):
    write_bidsignore(tmp_path)
    return (tmp_path / '.bidsignore').read_text().splitlines()


@pytest.mark.parametrize('path', OUTPUTS)
def test_bidsignore_covers_outputs(patterns, path):
    """Each output BIDS does not define is ignored (#1404)."""
    assert _ignored(path, patterns)


def test_bidsignore_keeps_bids_files(patterns):
    """Files BIDS defines are still validated."""
    assert not _ignored('dataset_description.json', patterns)
