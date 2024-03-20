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
"""Utilities and mocks for testing and documentation building."""
from contextlib import contextmanager
from pathlib import Path
from tempfile import mkdtemp

from niworkflows.data import Loader
from toml import loads

_load_data = Loader('mriqc')


@contextmanager
def mock_config():
    """Create a mock config for documentation and testing purposes."""
    from . import config

    filename = _load_data('data/config-example.toml')
    settings = loads(filename.read_text())
    for sectionname, configs in settings.items():
        if sectionname != 'environment':
            section = getattr(config, sectionname)
            section.load(configs, init=False)
    config.nipype.init()
    config.loggers.init()

    config.execution.work_dir = Path(mkdtemp())
    config.execution.bids_dir = Path(_load_data('data/tests/ds000005')).absolute()
    config.execution.init()

    yield
