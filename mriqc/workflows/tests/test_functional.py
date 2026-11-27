# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# Copyright 2026 The NiPreps Developers <nipreps@gmail.com>
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
"""Regression tests for functional workflow IQM collection."""

from pathlib import Path

import nipype.interfaces.utility as niu
import nipype.pipeline.engine as pe
import pandas as pd
import pytest

from mriqc import config
from mriqc.testing import mock_config
from mriqc.workflows.functional.base import _parse_tout, _parse_tqual, compute_iqms
from mriqc.workflows.utils import _tofloat


@pytest.mark.parametrize('n_echos', [1, 3])
@pytest.mark.parametrize('fft_spikes', [False, True])
def test_compute_iqms_datasink_iterfields(tmp_path, monkeypatch, n_echos, fft_spikes):
    """Multi-echo BOLD writes scalar per-echo aor, aqi, gcor, and spikes_num (gh-1386)."""
    bids_dir = tmp_path / 'bids' / 'sub-01' / 'func'
    bids_dir.mkdir(parents=True)
    out_dir = tmp_path / 'out'
    out_dir.mkdir()

    in_files, out_files, qual_files = [], [], []
    for idx in range(1, n_echos + 1):
        fname = (
            f'sub-01_task-rest_echo-{idx}_bold.nii.gz'
            if n_echos > 1
            else 'sub-01_task-rest_bold.nii.gz'
        )
        bold = bids_dir / fname
        bold.touch()
        in_files.append(str(bold))

        tout = tmp_path / f'outliers_{idx}.out'
        tout.write_text(f'{0.01 * idx}\n{0.02 * idx}\n')
        out_files.append(str(tout))

        tqual = tmp_path / f'quality_{idx}.out'
        tqual.write_text(f'{0.10 * idx}\n{0.20 * idx}\n')
        qual_files.append(str(tqual))

    with mock_config():
        monkeypatch.setattr(config.execution, 'output_dir', out_dir)
        monkeypatch.setattr(config.workflow, 'biggest_file_gb', {'bold': 0.1})
        monkeypatch.setattr(config.workflow, 'fft_spikes_detector', fft_spikes)
        iqms_wf = compute_iqms()
        datasink = iqms_wf.get_node('datasink')

    assert {'aor', 'gcor', 'aqi'}.issubset(datasink.iterfield)
    if fft_spikes:
        assert 'spikes_num' in datasink.iterfield

    wf = pe.Workflow('test_datasink', base_dir=str(tmp_path / 'work'))
    inp = pe.Node(
        niu.IdentityInterface(
            fields=[
                'in_file',
                'root',
                'metadata',
                'provenance',
                'outliers',
                'quality',
                'gcor',
                'spikes',
            ]
        ),
        name='inp',
    )
    inp.inputs.in_file = in_files if n_echos > 1 else in_files[0]
    inp.inputs.root = [{'snr': 10.0 * idx} for idx in range(1, n_echos + 1)]
    inp.inputs.metadata = (
        [{'EchoTime': 0.015 * idx} for idx in range(1, n_echos + 1)]
        if n_echos > 1
        else {'EchoTime': 0.015}
    )
    inp.inputs.provenance = [{'md5sum': f'hash{idx}'} for idx in range(1, n_echos + 1)]
    inp.inputs.outliers = out_files
    inp.inputs.quality = qual_files
    inp.inputs.gcor = [0.001 * idx for idx in range(1, n_echos + 1)]
    inp.inputs.spikes = list(range(1, n_echos + 1))

    connections = [
        ('in_file', 'in_file'),
        ('root', 'root'),
        ('metadata', 'metadata'),
        ('provenance', 'provenance'),
        (('outliers', _parse_tout), 'aor'),
        (('gcor', _tofloat), 'gcor'),
        (('quality', _parse_tqual), 'aqi'),
    ]
    if fft_spikes:
        connections.append(('spikes', 'spikes_num'))
    wf.connect([(inp, datasink, connections)])
    wf.run()

    parquets = sorted(Path(out_dir).rglob('*+iqms.parquet'))
    assert len(parquets) == n_echos
    for idx, pq in enumerate(parquets, start=1):
        df = pd.read_parquet(pq)
        assert df['aor'].iloc[0] == pytest.approx(0.015 * idx)
        assert df['aqi'].iloc[0] == pytest.approx(0.15 * idx)
        assert df['gcor'].iloc[0] == pytest.approx(0.001 * idx)
        assert df['snr'].iloc[0] == pytest.approx(10.0 * idx)
        if fft_spikes:
            assert df['spikes_num'].iloc[0] == idx
