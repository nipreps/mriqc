# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Reports."""
import numpy as np
import nibabel as nb
from nipype.interfaces.base import (
    traits,
    TraitedSpec,
    File,
    isdefined,
    InputMultiObject,
    BaseInterfaceInputSpec,
    SimpleInterface,
)
from .. import config
from ..reports.individual import individual_html


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
            "settings": {"testing": config.execution.debug, },
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
                    "hmc_fsl": config.workflow.hmc_fsl,
                }
            )

        return runtime
