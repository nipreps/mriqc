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
"""Exercise report generation with *NiReports*."""

from json import loads
from pathlib import Path

import pytest
from nireports.assembler.report import Report


@pytest.mark.parametrize(
    ('dataset', 'subject'),
    [
        ('ds002785', '0017'),
        ('ds002785', '0042'),
    ],
)
def test_anat_reports(tmp_path, testdata_path, outdir, dataset, subject):
    """Generate anatomical reports."""

    outdir = outdir if outdir is not None else tmp_path
    mriqc_data = Path(__file__).parent.parent / 'data'
    reportlets_dir = mriqc_data / 'tests' / dataset

    mriqc_json = loads(
        (reportlets_dir / f'sub-{subject}' / 'anat' / f'sub-{subject}_T1w.json').read_text()
    )

    Report(
        outdir,
        bootstrap_file=mriqc_data / 'bootstrap-anat.yml',
        run_uuid='some-id',
        reportlets_dir=reportlets_dir,
        subject=subject,
        suffix='T1w',
        metadata={
            'dataset': dataset,
            'about-metadata': {
                'Provenance Information': mriqc_json.pop('provenance'),
                'Dataset Information': mriqc_json.pop('bids_meta'),
                'Extracted Image quality metrics (IQMs)': mriqc_json,
            },
        },
        plugin_meta={
            'filename': f'sub-{subject}_T1w.nii.gz',
            'dataset': dataset,
        },
    ).generate_report()
