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
"""
The workflow builder factory method.

All the checks and the construction of the workflow are done
inside this function that has pickleable inputs and output
dictionary (``retval``) to allow isolation using a
``multiprocessing.Process`` that allows dmriprep to enforce
a hard-limited memory-scope.
"""


def build_workflow(config_file, retval):
    """Create the Nipype Workflow that supports the whole execution graph."""
    import os

    from mriqc import config

    # We do not need OMP > 1 for workflow creation
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['NUMEXPR_MAX_THREADS'] = '1'

    from mriqc.workflows.core import init_mriqc_wf

    config.load(config_file)
    # Initialize nipype config
    config.nipype.init()
    # Make sure loggers are started
    config.loggers.init()

    retval['return_code'] = 1
    retval['workflow'] = None

    config.loggers.cli.log(25, "Building MRIQC's workflows...")
    retval['workflow'] = init_mriqc_wf()
    retval['return_code'] = int(retval['workflow'] is None)
    config.loggers.cli.log(25, f'Workflow building finished (exit code {retval["return_code"]}).')
    return retval
