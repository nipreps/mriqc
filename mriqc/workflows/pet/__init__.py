"""PET quality control workflows.

This submodule defines Nipype workflows used by MRIQC to compute image
quality metrics (IQMs) and generate reportlets for PET datasets.  The
:func:`pet_qc_workflow` function orchestrates motion correction,
spatial normalization, TAC extraction and IQM computation, calling the
other constructors in this module.  Reportlets can be generated with
:func:`init_pet_report_wf`.
"""

from .base import (
    pet_qc_workflow,
    hmc,
    compute_iqms,
    pet_mni_align,
    extract_tacs,
)
from .output import init_pet_report_wf

__all__ = [
    'pet_qc_workflow',
    'hmc',
    'compute_iqms',
    'pet_mni_align',
    'extract_tacs',
    'init_pet_report_wf',
]