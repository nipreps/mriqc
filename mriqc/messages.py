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
"""Logging and console messages."""

BIDS_META = "Generating BIDS derivatives metadata."
BUILDING_WORKFLOW = "Building anatomical MRIQC workflow {detail}."
CREATED_DATASET = (
    'Created dataset X="{feat_file}", Y="{label_file}" (N={n_samples} valid samples)'
)
DROPPING_NON_NUMERICAL = "Dropping {n_labels} samples for having non-numerical labels"
GROUP_FINISHED = "Group level finished successfully."
GROUP_NO_DATA = "No data found. No group level reports were generated."
GROUP_START = "Group level started..."
INDIVIDUAL_REPORT_GENERATED = "Generated individual log: {out_file}"
PARTICIPANT_START = """
    Running MRIQC version {version}:
      * BIDS dataset path: {bids_dir}.
      * Output folder: {output_dir}.
      * Analysis levels: {analysis_level}.
"""
PARTICIPANT_FINISHED = "Participant level finished successfully."
POST_Z_NANS = "Columns {nan_columns} contain NaNs after z-scoring."
QC_UPLOAD_COMPLETE = "QC metrics successfully uploaded."
QC_UPLOAD_START = "MRIQC Web API: submitting to <{url}>"
GROUP_REPORT_GENERATED = "Group-{modality} report generated: {path}"
RUN_FINISHED = "MRIQC completed."
SUSPICIOUS_DATA_TYPE = "Input image {in_file} has a suspicious data type: '{dtype}'"
TSV_GENERATED = "Generated summary TSV table for {modality} data: {path}"
VOXEL_SIZE_OK = "Voxel size is large enough."
VOXEL_SIZE_SMALL = (
    "One or more voxel dimensions (%f, %f, %f) are smaller than the "
    "requested voxel size (%f) - diff=(%f, %f, %f)"
)
Z_SCORING = "z-scoring dataset..."
