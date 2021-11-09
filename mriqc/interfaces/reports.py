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
"""Reports."""
import nibabel as nb
import numpy as np
from mriqc import config
from mriqc.reports.individual import individual_html
from nipype.interfaces.base import (
    BaseInterfaceInputSpec,
    File,
    InputMultiObject,
    SimpleInterface,
    TraitedSpec,
    isdefined,
    traits,
)


class _IndividualReportInputSpec(BaseInterfaceInputSpec):
    in_iqms = File(exists=True)
    in_plots = InputMultiObject(File(exists=True))
    api_id = traits.Str()


class _IndividualReportOutputSpec(TraitedSpec):
    out_file = File(exists=True)


class IndividualReport(SimpleInterface):
    """Builds a provenance dictionary."""

    input_spec = _IndividualReportInputSpec
    output_spec = _IndividualReportOutputSpec

    def _run_interface(self, runtime):
        self._results["out_file"] = individual_html(
            self.inputs.in_iqms,
            self.inputs.in_plots if isdefined(self.inputs.in_plots) else None,
            self.inputs.api_id if isdefined(self.inputs.api_id) else None,
        )
        return runtime


class _AddProvenanceInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, desc="input file")
    air_msk = File(exists=True, desc="air mask file")
    rot_msk = File(exists=True, desc="rotation mask file")
    modality = traits.Str(mandatory=True, desc="provenance type")


class _AddProvenanceOutputSpec(TraitedSpec):
    out_prov = traits.Dict()


class AddProvenance(SimpleInterface):
    """Builds a provenance dictionary."""

    input_spec = _AddProvenanceInputSpec
    output_spec = _AddProvenanceOutputSpec

    def _run_interface(self, runtime):
        from nipype.utils.filemanip import hash_infile

        self._results["out_prov"] = {
            "md5sum": hash_infile(self.inputs.in_file),
            "version": config.environment.version,
            "software": "mriqc",
            "webapi_url": config.execution.webapi_url,
            "webapi_port": config.execution.webapi_port,
            "settings": {
                "testing": config.execution.debug,
            },
        }

        if self.inputs.modality in ("T1w", "T2w"):
            air_msk_size = (
                np.asanyarray(nb.load(self.inputs.air_msk).dataobj).astype(bool).sum()
            )
            rot_msk_size = (
                np.asanyarray(nb.load(self.inputs.rot_msk).dataobj).astype(bool).sum()
            )
            self._results["out_prov"]["warnings"] = {
                "small_air_mask": bool(air_msk_size < 5e5),
                "large_rot_frame": bool(rot_msk_size > 500),
            }

        if self.inputs.modality == "bold":
            self._results["out_prov"]["settings"].update(
                {
                    "fd_thres": config.workflow.fd_thres,
                }
            )

        return runtime
