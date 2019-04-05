#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: oesteban
# @Date:   2015-11-19 16:44:27
# @Last Modified by:   oesteban
# @Last Modified time: 2018-03-12 11:49:52

"""
MRIQC Plot script

"""

import os
import os.path as op
from argparse import ArgumentParser
from argparse import RawTextHelpFormatter

from .. import __version__
from ..reports import workflow_report


def main():
    """Entry point"""
    parser = ArgumentParser(description='MRI Quality Control',
                            formatter_class=RawTextHelpFormatter)

    g_input = parser.add_argument_group('Inputs')
    g_input.add_argument('-d', '--data-type', action='store', nargs='*',
                         choices=['anat', 'func'], default=['anat', 'func'])
    g_input.add_argument('-v', '--version', action='store_true', default=False,
                         help='Show current mriqc version')

    g_input.add_argument('--nthreads', action='store', default=0,
                         type=int, help='number of threads')

    g_outputs = parser.add_argument_group('Outputs')
    g_outputs.add_argument('-o', '--output-dir', action='store')
    g_outputs.add_argument('-w', '--work-dir', action='store',
                           default=op.join(os.getcwd(), 'work'))

    opts = parser.parse_args()
    if opts.version:
        print('mriqc version ' + __version__)
        exit(0)

    settings = {'output_dir': os.getcwd(),
                'nthreads': opts.nthreads}

    if opts.output_dir:
        settings['output_dir'] = op.abspath(opts.output_dir)

    if not op.exists(settings['output_dir']):
        os.makedirs(settings['output_dir'])

    settings['work_dir'] = op.abspath(opts.work_dir)
    if not op.exists(settings['work_dir']):
        raise RuntimeError('Work directory of a previous MRIQC run was not found.')

    for dtype in opts.data_type:
        workflow_report(dtype, settings)


if __name__ == '__main__':
    main()
