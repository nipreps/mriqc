#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: oesteban
# @Date:   2016-03-16 11:28:27
# @Last Modified by:   oesteban
# @Last Modified time: 2016-03-16 15:12:13

"""
Agave app generator

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import json
from argparse import ArgumentParser, RawTextHelpFormatter
import mriqc

agaveapp = {
    'name': mriqc.__name__,
    'version': mriqc.__versionbase__,
    'helpURI': 'http://mriqc.readthedocs.org',
    'label': 'mriqc',
    'shortDescription': mriqc.__description__,
    'longDescription': mriqc.__longdesc__,
    'executionSystem': 'slurm-ls5.tacc.utexas.edu',
    'executionType': 'HPC',
    'parallelism': 'PARALLEL',
    'defaultQueue': 'normal',
    'defaultNodeCount': 1,
    'deploymentPath': 'apps/mriqc',
    'deploymentSystem': 'openfmri-storage',
    'templatePath': 'wrapper.sh',
    'testPath': 'test/test.sh',
    'tags': ['qc', 'sMRI', 'fMRI'],
    'modules': [],
    'inputs': [{
        'id': 'bidsFolder',
        'details': {
            'label': 'folder',
            'description': 'input root folder of a BIDS-compliant tree',
            'argument': None,
            'showArgument': False
        },
        'value': {
            'visible': True,
            'required': True,
            'type': 'string',
            'default': 'agave://openfmri-storage/data/ds003_downsampled'
        },
        'semantics': {
            'ontology': []
        }
    }],
    'parameters': [],
    'checkpointable': False
}

def main():
    """Entry point"""

    parser = ArgumentParser(description='ABIDE2BIDS downloader',
                            formatter_class=RawTextHelpFormatter)
    g_inputs = parser.add_argument_group('Inputs')
    g_inputs.add_argument('--help-uri', action='store')
    g_inputs.add_argument('--input-name', action='store')
    g_inputs.add_argument('-E', '--exec-system', action='store')


    g_outputs = parser.add_argument_group('Outputs')
    g_outputs.add_argument('-o', '--output', action='store',
                           default='app.json')

    opts = parser.parse_args()

    if opts.input_name is not None:
        agaveapp['inputs'][0]['id'] = opts.input_name
    if opts.exec_system:
        agaveapp['executionSystem'] = opts.exec_system

    with open(opts.output, 'w') as appfile:
        json.dump(agaveapp, appfile, indent=4)



if __name__ == '__main__':
    main()
