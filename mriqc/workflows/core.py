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
Combines the structural and functional MRI workflows.
"""
from mriqc.workflows.anatomical import anat_qc_workflow
from mriqc.workflows.functional import fmri_qc_workflow
from nipype.pipeline.engine import Workflow

ANATOMICAL_KEYS = "T1w", "T2w"
FMRI_KEY = "bold"


def init_mriqc_wf():
    """Create a multi-subject MRIQC workflow."""
    from mriqc import config

    # Create parent workflow
    workflow = Workflow(name="mriqc_wf")
    workflow.base_dir = config.execution.work_dir

    # Create fMRI QC workflow
    if FMRI_KEY in config.workflow.inputs:
        workflow.add_nodes([fmri_qc_workflow()])

    # Create sMRI QC workflow
    input_keys = config.workflow.inputs.keys()
    anatomical_flag = any(key in input_keys for key in ANATOMICAL_KEYS)
    if anatomical_flag:
        workflow.add_nodes([anat_qc_workflow()])

    # Return non-empty workflow, else None
    if workflow._get_all_nodes():
        return workflow
