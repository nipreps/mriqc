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
"""Tests for uploading the parquet IQMs to the MRIQC Web API."""

import logging
from types import SimpleNamespace

import orjson
import requests

from mriqc import config
from mriqc.interfaces.bids import IQMFileSink
from mriqc.interfaces.webapi import upload_qc_metrics


def test_upload_keeps_integer_iqms(tmp_path, monkeypatch):
    in_file = tmp_path / 'ds' / 'sub-01' / 'anat' / 'sub-01_T1w.nii.gz'
    in_file.parent.mkdir(parents=True)
    in_file.touch()

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(config.loggers, 'interface', logging.getLogger('nipype.interface'))
    sink = IQMFileSink(
        in_file=str(in_file),
        modality='T1w',
        # As the workflows pass them, from ``BIDSLayout.parse_file_entities``
        entities={'subject': '01', 'datatype': 'anat', 'suffix': 'T1w', 'extension': '.nii.gz'},
        out_dir=str(tmp_path / 'out'),
        root={'size_x': 256, 'size_y': 256, 'size_z': 176, 'cjv': 0.42},
        provenance={'version': '26.0.0'},
    )
    out_file = sink.run().outputs.out_file

    sent = {}

    def _post(url, headers=None, data=None, timeout=None):
        sent['url'] = url
        sent['data'] = orjson.loads(data)
        return SimpleNamespace(status_code=201, text='', json=lambda: {'_id': 'x'})

    monkeypatch.setattr(requests, 'post', _post)
    response, _ = upload_qc_metrics(
        out_file,
        endpoint='http://api',
        auth_token='not-a-secret',  # noqa: S106
    )

    assert response.status_code == 201
    assert sent['url'] == 'http://api/T1w'
    # The Web API rejects non-integer image sizes
    assert all(type(sent['data'][f'size_{ax}']) is int for ax in 'xyz')
    assert sent['data']['cjv'] == 0.42
