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
from builtins import object  # pylint: disable=W0622
from io import open
from sys import version_info

import pandas as pd

from .. import config
from ..utils.misc import BIDS_COMP


def gen_html(csv_file, mod, csv_failed=None, out_file=None):
    import datetime
    import os.path as op

    from pkg_resources import resource_filename as pkgrf

    from .. import __version__ as ver
    from ..data import GroupTemplate

    if version_info[0] > 2:
        from io import StringIO as TextIO
    else:
        from io import BytesIO as TextIO

    QCGROUPS = {
        "T1w": [
            (["cjv"], None),
            (["cnr"], None),
            (["efc"], None),
            (["fber"], None),
            (["wm2max"], None),
            (["snr_csf", "snr_gm", "snr_wm"], None),
            (["snrd_csf", "snrd_gm", "snrd_wm"], None),
            (["fwhm_avg", "fwhm_x", "fwhm_y", "fwhm_z"], "vox"),
            (["qi_1", "qi_2"], None),
            (["inu_range", "inu_med"], None),
            (["icvs_csf", "icvs_gm", "icvs_wm"], None),
            (["rpve_csf", "rpve_gm", "rpve_wm"], None),
            (["tpm_overlap_csf", "tpm_overlap_gm", "tpm_overlap_wm"], None),
            (
                [
                    "summary_bg_mean",
                    "summary_bg_median",
                    "summary_bg_stdv",
                    "summary_bg_mad",
                    "summary_bg_k",
                    "summary_bg_p05",
                    "summary_bg_p95",
                ],
                None,
            ),
            (
                [
                    "summary_csf_mean",
                    "summary_csf_median",
                    "summary_csf_stdv",
                    "summary_csf_mad",
                    "summary_csf_k",
                    "summary_csf_p05",
                    "summary_csf_p95",
                ],
                None,
            ),
            (
                [
                    "summary_gm_mean",
                    "summary_gm_median",
                    "summary_gm_stdv",
                    "summary_gm_mad",
                    "summary_gm_k",
                    "summary_gm_p05",
                    "summary_gm_p95",
                ],
                None,
            ),
            (
                [
                    "summary_wm_mean",
                    "summary_wm_median",
                    "summary_wm_stdv",
                    "summary_wm_mad",
                    "summary_wm_k",
                    "summary_wm_p05",
                    "summary_wm_p95",
                ],
                None,
            ),
        ],
        "T2w": [
            (["cjv"], None),
            (["cnr"], None),
            (["efc"], None),
            (["fber"], None),
            (["wm2max"], None),
            (["snr_csf", "snr_gm", "snr_wm"], None),
            (["snrd_csf", "snrd_gm", "snrd_wm"], None),
            (["fwhm_avg", "fwhm_x", "fwhm_y", "fwhm_z"], "mm"),
            (["qi_1", "qi_2"], None),
            (["inu_range", "inu_med"], None),
            (["icvs_csf", "icvs_gm", "icvs_wm"], None),
            (["rpve_csf", "rpve_gm", "rpve_wm"], None),
            (["tpm_overlap_csf", "tpm_overlap_gm", "tpm_overlap_wm"], None),
            (
                [
                    "summary_bg_mean",
                    "summary_bg_stdv",
                    "summary_bg_k",
                    "summary_bg_p05",
                    "summary_bg_p95",
                ],
                None,
            ),
            (
                [
                    "summary_csf_mean",
                    "summary_csf_stdv",
                    "summary_csf_k",
                    "summary_csf_p05",
                    "summary_csf_p95",
                ],
                None,
            ),
            (
                [
                    "summary_gm_mean",
                    "summary_gm_stdv",
                    "summary_gm_k",
                    "summary_gm_p05",
                    "summary_gm_p95",
                ],
                None,
            ),
            (
                [
                    "summary_wm_mean",
                    "summary_wm_stdv",
                    "summary_wm_k",
                    "summary_wm_p05",
                    "summary_wm_p95",
                ],
                None,
            ),
        ],
        "bold": [
            (["efc"], None),
            (["fber"], None),
            (["fwhm", "fwhm_x", "fwhm_y", "fwhm_z"], "mm"),
            (["gsr_%s" % a for a in ["x", "y"]], None),
            (["snr"], None),
            (["dvars_std", "dvars_vstd"], None),
            (["dvars_nstd"], None),
            (["fd_mean"], "mm"),
            (["fd_num"], "# timepoints"),
            (["fd_perc"], "% timepoints"),
            (["spikes_num"], "# slices"),
            (["dummy_trs"], "# TRs"),
            (["gcor"], None),
            (["tsnr"], None),
            (["aor"], None),
            (["aqi"], None),
            (
                [
                    "summary_bg_mean",
                    "summary_bg_stdv",
                    "summary_bg_k",
                    "summary_bg_p05",
                    "summary_bg_p95",
                ],
                None,
            ),
            (
                [
                    "summary_fg_mean",
                    "summary_fg_stdv",
                    "summary_fg_k",
                    "summary_fg_p05",
                    "summary_fg_p95",
                ],
                None,
            ),
        ],
    }

    if csv_file.suffix == ".csv":
        def_comps = list(BIDS_COMP.keys())
        dataframe = pd.read_csv(
            csv_file, index_col=False, dtype={comp: object for comp in def_comps}
        )

        id_labels = list(set(def_comps) & set(dataframe.columns.ravel().tolist()))
        dataframe["label"] = dataframe[id_labels].apply(
            _format_labels, args=(id_labels,), axis=1
        )
    else:
        dataframe = pd.read_csv(
            csv_file, index_col=False, sep="\t", dtype={"bids_name": object}
        )
        dataframe = dataframe.rename(index=str, columns={"bids_name": "label"})

    nPart = len(dataframe)

    failed = None
    if csv_failed is not None and op.isfile(csv_failed):
        config.loggers.cli.warning(f'Found failed-workflows table "{csv_failed}"')
        failed_df = pd.read_csv(csv_failed, index_col=False)
        cols = list(set(id_labels) & set(failed_df.columns.ravel().tolist()))

        try:
            failed_df = failed_df.sort_values(by=cols)
        except AttributeError:
            failed_df = failed_df.sort(columns=cols)

        # myfmt not defined
        # failed = failed_df[cols].apply(myfmt, args=(cols,), axis=1).ravel().tolist()

    csv_groups = []
    datacols = dataframe.columns.ravel().tolist()
    for group, units in QCGROUPS[mod]:
        dfdict = {"iqm": [], "value": [], "label": [], "units": []}

        for iqm in group:
            if iqm in datacols:
                values = dataframe[[iqm]].values.ravel().tolist()
                if values:
                    dfdict["iqm"] += [iqm] * nPart
                    dfdict["units"] += [units] * nPart
                    dfdict["value"] += values
                    dfdict["label"] += dataframe[["label"]].values.ravel().tolist()

        # Save only if there are values
        if dfdict["value"]:
            csv_df = pd.DataFrame(dfdict)
            csv_str = TextIO()
            csv_df[["iqm", "value", "label", "units"]].to_csv(csv_str, index=False)
            csv_groups.append(csv_str.getvalue())

    if out_file is None:
        out_file = op.abspath("group.html")
    tpl = GroupTemplate()
    tpl.generate_conf(
        {
            "modality": mod,
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d, %H:%M"),
            "version": ver,
            "csv_groups": csv_groups,
            "failed": failed,
            "boxplots_js": open(
                pkgrf(
                    "mriqc",
                    op.join("data", "reports", "embed_resources", "boxplots.js"),
                )
            ).read(),
            "d3_js": open(
                pkgrf(
                    "mriqc", op.join("data", "reports", "embed_resources", "d3.min.js")
                )
            ).read(),
            "boxplots_css": open(
                pkgrf(
                    "mriqc",
                    op.join("data", "reports", "embed_resources", "boxplots.css"),
                )
            ).read(),
        },
        out_file,
    )

    return out_file


def _format_labels(row, id_labels):
    """format participant labels"""
    crow = []

    for col_id, prefix in list(BIDS_COMP.items()):
        if col_id in id_labels:
            crow.append("%s-%s" % (prefix, row[[col_id]].values[0]))
    return "_".join(crow)
