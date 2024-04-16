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
"""py.test configuration"""

import os
import tempfile
from pathlib import Path
from sys import version_info

import nibabel as nb
import numpy as np
import pandas as pd
import pytest

# disable ET
os.environ['NO_ET'] = '1'

_datadir = (Path(__file__).parent / 'data' / 'tests').resolve(strict=True)
niprepsdev_path = os.getenv('TEST_DATA_HOME', str(Path.home() / '.cache' / 'mriqc'))
test_output_dir = os.getenv('TEST_OUTPUT_DIR')
test_workdir = os.getenv('TEST_WORK_DIR')


@pytest.fixture(autouse=True)
def expand_namespace(doctest_namespace):
    doctest_namespace['PY_VERSION'] = version_info
    doctest_namespace['np'] = np
    doctest_namespace['nb'] = nb
    doctest_namespace['pd'] = pd
    doctest_namespace['os'] = os
    doctest_namespace['pytest'] = pytest
    doctest_namespace['Path'] = Path
    doctest_namespace['testdata_path'] = _datadir
    doctest_namespace['niprepsdev_path'] = niprepsdev_path

    doctest_namespace['os'] = os
    doctest_namespace['Path'] = Path

    tmpdir = tempfile.TemporaryDirectory()
    doctest_namespace['tmpdir'] = tmpdir.name

    doctest_namespace['output_dir'] = (
        Path(test_output_dir) if test_output_dir is not None else Path(tmpdir.name)
    )

    cwd = os.getcwd()
    os.chdir(tmpdir.name)
    yield
    os.chdir(cwd)
    tmpdir.cleanup()


@pytest.fixture()
def testdata_path():
    return _datadir


@pytest.fixture()
def workdir():
    return None if test_workdir is None else Path(test_workdir)


@pytest.fixture()
def outdir():
    return None if test_output_dir is None else Path(test_output_dir)
