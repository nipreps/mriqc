# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# Copyright 2023 The NiPreps Developers <nipreps@gmail.com>
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
"""Interfaces for manipulating DWI data."""
import numpy as np
from nipype.interfaces.base import File, traits
from niworkflows.interfaces.bids import ReadSidecarJSON, _ReadSidecarJSONOutputSpec


class _ReadDWIMetadataOutputSpec(_ReadSidecarJSONOutputSpec):
    out_bvec_file = File(desc="corresponding bvec file")
    out_bval_file = File(desc="corresponding bval file")
    out_bmatrix = traits.List(traits.List(traits.Float), desc="b-matrix")


class ReadDWIMetadata(ReadSidecarJSON):
    """
    Extends the NiWorkflows' interface to extract bvec/bval from DWI datasets.
    """

    output_spec = _ReadDWIMetadataOutputSpec

    def _run_interface(self, runtime):
        runtime = super()._run_interface(runtime)

        self._results["out_bvec_file"] = str(self.layout.get_bvec(self.inputs.in_file))
        self._results["out_bval_file"] = str(self.layout.get_bval(self.inputs.in_file))

        bvecs = np.loadtxt(self._results["out_bvec_file"]).T
        bvals = np.loadtxt(self._results["out_bval_file"])

        self._results["out_bmatrix"] = np.hstack((bvecs, bvals[:, np.newaxis])).tolist()

        return runtime
