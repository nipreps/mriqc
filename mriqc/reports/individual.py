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

from json import loads
from pathlib import Path

from nireports.assembler.report import Report
from niworkflows.data import Loader

_load_data = Loader('mriqc')


def generate_reports():
    """Generate the reports associated with an MRIQC run."""

    from mriqc import config

    config.loggers.workflow.info('Generating reports...')
    output_files = [_single_report(ff) for mod in config.workflow.inputs.values() for ff in mod]
    config.loggers.workflow.info(f'Report generation finished ({len(output_files)} reports).')
    return output_files


def _single_report(in_file):
    """Generate a single report."""
    from mriqc import config

    # Ensure it's a Path
    in_file = Path(in_file if not isinstance(in_file, list) else in_file[0])

    # Extract BIDS entities
    entities = config.execution.layout.get_file(in_file).get_entities()
    entities.pop('extension', None)
    entities.pop('echo', None)
    entities.pop('part', None)
    report_type = entities.pop('datatype', None)

    # Read output file:
    mriqc_json = loads(
        (
            Path(config.execution.output_dir)
            / in_file.parent.relative_to(config.execution.bids_dir)
            / in_file.name.replace(''.join(in_file.suffixes), '.json')
        ).read_text()
    )
    mriqc_json.pop('bids_meta')

    # Clean-up provenance dictionary
    prov = mriqc_json.pop('provenance', None)
    prov.pop('webapi_url', None)
    prov.pop('webapi_port', None)
    prov.pop('settings', None)
    prov.pop('software', None)
    prov.update({f'warnings_{kk}': vv for kk, vv in prov.pop('warnings', {}).items()})
    prov['Input filename'] = f'<BIDS root>/{in_file.relative_to(config.execution.bids_dir)}'
    prov['Versions_MRIQC'] = prov.pop('version', config.environment.version)
    prov['Execution environment'] = config.environment.exec_env
    prov['Versions_NiPype'] = config.environment.nipype_version
    prov['Versions_TemplateFlow'] = config.environment.templateflow_version

    bids_meta = config.execution.layout.get_file(in_file).get_metadata()
    bids_meta.pop('global', None)

    robj = Report(
        config.execution.output_dir,
        config.execution.run_uuid,
        reportlets_dir=config.execution.work_dir / 'reportlets',
        bootstrap_file=_load_data(f'data/bootstrap-{report_type}.yml'),
        metadata={
            'dataset': config.execution.dsname,
            'about-metadata': {
                'Provenance Information': prov,
                'Dataset Information': bids_meta,
                'Extracted Image quality metrics (IQMs)': mriqc_json,
            },
        },
        plugin_meta={
            'rating-widget': {
                'filename': in_file.name,
                'dataset': config.execution.dsname,
                'access_token': config.execution.webapi_token,
                'endpoint': f'{config.execution.webapi_url}/rating',
            },
        },
        **entities,
    )
    robj.generate_report()
    return robj.out_filename.absolute()
