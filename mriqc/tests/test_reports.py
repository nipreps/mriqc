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

import re
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


@pytest.mark.parametrize('acquisition', [None, 'hippo'])
def test_report_entities(tmp_path, monkeypatch, acquisition):
    """A report only shows the figures of its own input (gh-1303)."""
    import pandas as pd
    from bids.layout import BIDSLayout

    from mriqc import config
    from mriqc.reports.individual import _single_report

    bids_dir = tmp_path / 'ds'
    out_dir = tmp_path / 'out'
    figures_dir = tmp_path / 'work' / 'reportlets' / 'sub-01' / 'figures'
    for path in (bids_dir / 'sub-01' / 'anat', out_dir / 'sub-01' / 'anat', figures_dir):
        path.mkdir(parents=True)
    (bids_dir / 'dataset_description.json').write_text(
        '{"Name": "Example dataset", "BIDSVersion": "1.10.0"}'
    )

    for acq in ('', 'acq-hippo_'):
        (bids_dir / 'sub-01' / 'anat' / f'sub-01_{acq}T2w.nii.gz').write_text('')
        iqms_json = out_dir / 'sub-01' / 'anat' / f'sub-01_{acq}T2w+iqms.json'
        iqms_json.write_text('{"provenance": {}}')
        pd.DataFrame([{'cjv': 0.5}]).to_parquet(iqms_json.with_suffix('.parquet'))
        (figures_dir / f'sub-01_{acq}desc-zoomed_T2w.svg').write_text(
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1 1"></svg>'
        )

    monkeypatch.setattr(config.execution, 'bids_dir', bids_dir)
    monkeypatch.setattr(config.execution, 'output_dir', out_dir)
    monkeypatch.setattr(config.execution, 'work_dir', tmp_path / 'work')
    monkeypatch.setattr(config.execution, 'layout', BIDSLayout(bids_dir, validate=False))

    acq = f'acq-{acquisition}_' if acquisition else ''
    report = _single_report(bids_dir / 'sub-01' / 'anat' / f'sub-01_{acq}T2w.nii.gz')
    assert report == out_dir / f'sub-01_{acq}T2w.html'

    figures = sorted(set(re.findall(r'sub-01_[^/"]*\.svg', report.read_text())))
    assert figures == [f'sub-01_{acq}desc-zoomed_T2w.svg']


@pytest.mark.parametrize(
    ('mod', 'table'), [('T1w', 'T1w'), ('T2w', 'FLAIR'), ('FLAIR', 'FLAIR'), ('bold', 'bold')]
)
def test_group_report(tmp_path, mod, table):
    """Group reports plot every FWHM, in voxels (gh-835)."""
    from mriqc.reports.group import gen_html

    tsv = Path(__file__).parent.parent / 'data' / 'testdata' / f'group_{table}.tsv'
    html = Path(gen_html(tsv, mod, out_file=str(tmp_path / 'group.html'))).read_text()

    fwhm = [line.strip().split(',') for line in html.splitlines() if line.startswith('fwhm_')]
    assert {row[0] for row in fwhm} == {'fwhm_avg', 'fwhm_x', 'fwhm_y', 'fwhm_z'}
    assert {row[-1] for row in fwhm} == {'vox'}
