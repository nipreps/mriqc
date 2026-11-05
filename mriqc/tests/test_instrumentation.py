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
"""Check the resource-monitor plot (``--resource-monitor``)."""

from mriqc.instrumentation.resources import SAMPLE_ATTRS
from mriqc.instrumentation.viz import plot


def test_plot(tmp_path):
    """Memory samples are floats; pandas 3 refuses to add them into an integer column."""
    columns = ['timestamp', *SAMPLE_ATTRS]
    columns[columns.index('memory_info') : columns.index('memory_info') + 1] = [
        'mem_rss_mb',
        'mem_vsm_mb',
    ]
    rows = []
    for t in range(4):
        for pid, name in ((1, 'mriqc'), (2, 'antsRegistration')):
            sample = dict.fromkeys(columns, 1)
            sample.update(
                timestamp=t * 1_000_000_000, pid=pid, name=name, mem_vsm_mb=100.5 * (t + 1)
            )
            rows.append('\t'.join(str(sample[c]) for c in columns))
    log = tmp_path / 'resources.tsv'
    log.write_text('\n'.join(['# header', '\t'.join(columns), *rows]) + '\n')

    out_file = tmp_path / 'resources.png'
    plot(log, out_file=out_file)
    assert out_file.stat().st_size > 0
