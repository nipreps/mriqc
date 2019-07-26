# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
MRIQC - anatomical and functional workflows.

.. automodule:: mriqc.workflows.anatomical
    :members:
    :undoc-members:
    :show-inheritance:


.. automodule:: mriqc.workflows.functional
    :members:
    :undoc-members:
    :show-inheritance:


"""
from .anatomical import anat_qc_workflow
from .functional import fmri_qc_workflow

__all__ = [
    'anat_qc_workflow',
    'fmri_qc_workflow',
]
