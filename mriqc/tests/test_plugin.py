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
"""Tests for the execution plugin."""

import getpass
from pathlib import Path
from types import SimpleNamespace

from mriqc.engine.plugin import report_crash


def test_report_crash_unnamed_uid(tmp_path, monkeypatch):
    """Write the crash file when the UID has no user name, as in ``docker run -u``."""

    def _getuser():
        # What Python 3.13 raises; earlier versions raise KeyError
        raise OSError('No username set in the environment')

    monkeypatch.setattr(getpass, 'getuser', _getuser)
    node = SimpleNamespace(
        _id='node',
        fullname='wf.node',
        inputs='',
        output_dir=lambda: str(tmp_path),
        config={'execution': {'crashdump_dir': str(tmp_path), 'crashfile_format': 'txt'}},
    )

    crashfile = Path(report_crash(node, traceback=['Traceback (most recent call last):\n']))
    assert crashfile.is_file()
    assert '-UID' in crashfile.name
