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
"""PyBIDS tooling."""
import json
import os
from collections import defaultdict
from pathlib import Path

DEFAULT_TYPES = ["bold", "T1w", "T2w"]
DOI = "https://doi.org/10.1371/journal.pone.0184661"


def collect_bids_data(
    layout,
    participant_label=None,
    session=None,
    run=None,
    task=None,
    bids_type=None,
):
    """Get files in dataset"""

    bids_type = bids_type or DEFAULT_TYPES
    if not isinstance(bids_type, (list, tuple)):
        bids_type = [bids_type]

    basequery = {
        "subject": participant_label,
        "session": session,
        "task": task,
        "run": run,
        "datatype": "func",
        "return_type": "file",
        "extension": ["nii", "nii.gz"],
    }
    # Filter empty lists, strings, zero runs, and Nones
    basequery = {k: v for k, v in basequery.items() if v}

    # Start querying
    imaging_data = defaultdict(list, {})
    for btype in bids_type:
        _entities = basequery.copy()
        _entities["suffix"] = btype
        if btype in ("T1w", "T2w"):
            _entities["datatype"] = "anat"
            _entities.pop("task", None)

        imaging_data[btype] = layout.get(**_entities)

    return imaging_data


def write_bidsignore(deriv_dir):
    bids_ignore = (
        "*.html",
        "logs/",  # Reports
        "*_T1w.json",
        "*_T2w.json",
        "*_bold.json",  # Outputs are not yet standardized
    )
    ignore_file = Path(deriv_dir) / ".bidsignore"

    ignore_file.write_text("\n".join(bids_ignore) + "\n")


def write_derivative_description(bids_dir, deriv_dir):
    from mriqc import __download__, __version__

    bids_dir = Path(bids_dir)
    deriv_dir = Path(deriv_dir)
    desc = {
        "Name": "MRIQC - MRI Quality Control",
        "BIDSVersion": "1.4.0",
        "DatasetType": "derivative",
        "GeneratedBy": [
            {
                "Name": "MRIQC",
                "Version": __version__,
                "CodeURL": __download__,
            }
        ],
        "HowToAcknowledge": f"Please cite our paper ({DOI}).",
    }

    # Keys that can only be set by environment
    # XXX: This currently has no effect, but is a stand-in to remind us to figure out
    # how to detect the container
    if "MRIQC_DOCKER_TAG" in os.environ:
        desc["GeneratedBy"][0]["Container"] = {
            "Type": "docker",
            "Tag": f"nipreps/mriqc:{os.environ['MRIQC_DOCKER_TAG']}",
        }
    if "MRIQC_SINGULARITY_URL" in os.environ:
        desc["GeneratedBy"][0]["Container"] = {
            "Type": "singularity",
            "URI": os.getenv("MRIQC_SINGULARITY_URL"),
        }

    # Keys deriving from source dataset
    orig_desc = {}
    fname = bids_dir / "dataset_description.json"
    if fname.exists():
        orig_desc = json.loads(fname.read_text())

    if "DatasetDOI" in orig_desc:
        desc["SourceDatasets"] = [
            {
                "URL": f'https://doi.org/{orig_desc["DatasetDOI"]}',
                "DOI": orig_desc["DatasetDOI"],
            }
        ]
    if "License" in orig_desc:
        desc["License"] = orig_desc["License"]

    Path.write_text(deriv_dir / "dataset_description.json", json.dumps(desc, indent=4))
