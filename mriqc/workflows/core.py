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

    wf_list = []
    for mod, filelist in config.workflow.inputs.items():
        if mod == 'bold':
            wf_list.append(fmri_qc_workflow())
        else:
            wf_list.append(anat_qc_workflow())

    if not wf_list:
        return None

    workflow.add_nodes(wf_list)
    return workflow
