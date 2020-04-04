# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""The core module combines the existing workflows."""
from nipype.pipeline.engine import Workflow
from .anatomical import anat_qc_workflow
from .functional import fmri_qc_workflow


def init_mriqc_wf():
    """Create a multi-subject MRIQC workflow."""
    from .. import config

    workflow = Workflow(name="mriqc_wf")
    workflow.base_dir = config.execution.work_dir

    if "bold" in config.workflow.inputs:
        workflow.add_nodes([fmri_qc_workflow()])

    if set(("T1w", "T2w")).intersection(
        config.workflow.inputs.keys()
    ):
        workflow.add_nodes([anat_qc_workflow()])

    if not workflow._get_all_nodes():
        return None

    return workflow
