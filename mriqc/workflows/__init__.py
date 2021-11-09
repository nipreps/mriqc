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
.. automodule:: mriqc.workflows.anatomical
    :members:
    :undoc-members:
    :show-inheritance:


.. automodule:: mriqc.workflows.functional
    :members:
    :undoc-members:
    :show-inheritance:

"""
from mriqc.workflows.anatomical import anat_qc_workflow
from mriqc.workflows.functional import fmri_qc_workflow

__all__ = [
    "anat_qc_workflow",
    "fmri_qc_workflow",
]
