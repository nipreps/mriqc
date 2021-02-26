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
