"""
Logging and console messages.
"""

BIDS_META = "Generating BIDS derivatives metadata."
BUILDING_WORKFLOW = "Building anatomical MRIQC workflow for files: {dataset}"
CREATED_DATASET = 'Created dataset X="{feat_file}", Y="{label_file}" (N={n_samples} valid samples)'
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
VOXEL_SIZE_SMALL = "One or more voxel dimensions (%f, %f, %f) are smaller than the requested voxel size (%f) - diff=(%f, %f, %f)"
Z_SCORING = "z-scoring dataset..."

# flake8: noqa: E502
