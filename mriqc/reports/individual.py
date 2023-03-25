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
from pathlib import Path
from json import loads
from pkg_resources import resource_filename as pkgrf
from nireports.assembler.report import Report


def generate_reports():
    """Generate the reports associated with an MRIQC run."""

    from mriqc import config

    input_files = [
        Path(ff) for mod in config.workflow.inputs.values() for ff in mod
    ]

    for in_file in input_files:
        # Extract BIDS entities
        entities = config.execution.layout.get_file(in_file).entities
        entities.pop("extension")

        # Read output file:
        mriqc_json = loads((
            Path(config.execution.output_dir)
            / in_file.parent.relative_to(config.execution.bids_dir)
            / in_file.name.replace("".join(in_file.suffixes), ".json")
        ).read_text())

        Report(
            config.execution.output_dir,
            config.execution.run_uuid,
            reportlets_dir=config.execution.work_dir,
            bootstrap_file=pkgrf("mriqc", "data/report-bootstrap.yml"),
            metadata={
                "dataset": config.execution.dsname,
                "about-metadata": {
                    "Provenance Information": mriqc_json.pop("provenance"),
                    "Dataset Information": mriqc_json.pop("bids_meta"),
                    "Extracted Image quality metrics (IQMs)": mriqc_json,
                }
            },
            plugin_meta={
                "filename": in_file.name,
                "dataset": config.execution.dsname,
            },
            **entities,
        )
