# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Reports."""
from nipype.interfaces.base import (
    traits, TraitedSpec, File, isdefined, InputMultiObject,
    BaseInterfaceInputSpec, SimpleInterface
)
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
