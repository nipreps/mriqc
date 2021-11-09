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
"""Encapsulates report generation functions."""
from mriqc import messages


def individual_html(in_iqms, in_plots=None, api_id=None):
    import datetime
    from json import load
    from pathlib import Path

    from .. import config
    from ..data import IndividualTemplate
    from ..reports import REPORT_TITLES
    from ..reports.utils import iqms2html, read_report_snippet
    from ..utils.misc import BIDS_COMP

    def _get_details(in_iqms, modality):
        in_prov = in_iqms.pop("provenance", {})
        warn_dict = in_prov.pop("warnings", None)
        sett_dict = in_prov.pop("settings", None)

        wf_details = []
        if modality == "bold":
            bold_exclude_index = in_iqms.get("dumb_trs")
            if bold_exclude_index is None:
                config.loggers.cli.warning(
                    "Building bold report: no exclude index was found"
                )
            elif bold_exclude_index > 0:
                msg = """\
<span class="problematic">Non-steady state (strong T1 contrast) has been detected in the \
first {} volumes</span>. They were excluded before generating any QC measures and plots."""
                wf_details.append(msg.format(bold_exclude_index))

            wf_details.append(
                "Framewise Displacement was computed using <code>3dvolreg</code> (AFNI)"
            )

            fd_thres = sett_dict.pop("fd_thres")
            if fd_thres is not None:
                wf_details.append(
                    "Framewise Displacement threshold was defined at %f mm" % fd_thres
                )
        elif modality in ("T1w", "T2w"):
            if warn_dict.pop("small_air_mask", False):
                wf_details.append(
                    '<span class="problematic">Detected hat mask was too small</span>'
                )

            if warn_dict.pop("large_rot_frame", False):
                wf_details.append(
                    '<span class="problematic">Detected a zero-filled frame, has the original '
                    "image been rotated?</span>"
                )

        return in_prov, wf_details, sett_dict

    in_iqms = Path(in_iqms)
    with in_iqms.open() as jsonfile:
        iqms_dict = load(jsonfile)

    # Now, the in_iqms file should be correctly named
    out_file = str(Path(in_iqms.with_suffix(".html").name).resolve())

    # Extract and prune metadata
    metadata = iqms_dict.pop("bids_meta", None)
    mod = metadata.pop("modality", None)
    prov, wf_details, _ = _get_details(iqms_dict, mod)

    file_id = [metadata.pop(k, None) for k in list(BIDS_COMP.keys())]
    file_id = [comp for comp in file_id if comp is not None]

    if in_plots is None:
        in_plots = []
    else:
        if any(("melodic_reportlet" in k for k in in_plots)):
            REPORT_TITLES["bold"].insert(3, ("ICA components", "ica-comps"))
        if any(("plot_spikes" in k for k in in_plots)):
            REPORT_TITLES["bold"].insert(3, ("Spikes", "spikes"))

        in_plots = [
            (REPORT_TITLES[mod][i] + (read_report_snippet(v),))
            for i, v in enumerate(in_plots)
        ]

    pred_qa = None  # metadata.pop('mriqc_pred', None)
    _config = {
        "modality": mod,
        "dataset": metadata.pop("dataset", None),
        "bids_name": in_iqms.with_suffix("").name,
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d, %H:%M"),
        "version": config.environment.version,
        "imparams": iqms2html(iqms_dict, "iqms-table"),
        "svg_files": in_plots,
        "workflow_details": wf_details,
        "webapi_url": prov.pop("webapi_url"),
        "webapi_port": prov.pop("webapi_port"),
        "provenance": iqms2html(prov, "provenance-table"),
        "md5sum": prov["md5sum"],
        "metadata": iqms2html(metadata, "metadata-table"),
        "pred_qa": pred_qa,
    }

    if _config["metadata"] is None:
        _config["workflow_details"].append(
            '<span class="warning">File has no metadata</span> '
            "<span>(sidecar JSON file missing or empty)</span>"
        )

    tpl = IndividualTemplate()
    tpl.generate_conf(_config, out_file)

    end_message = messages.INDIVIDUAL_REPORT_GENERATED.format(out_file=out_file)
    config.loggers.cli.info(end_message)
    return out_file
