#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: oesteban
# @Date:   2015-11-19 16:44:27
# @Last Modified by:   oesteban
# @Last Modified time: 2016-05-06 11:14:22

"""
=====
MRIQC
=====
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import os.path as op
from warnings import warn
from multiprocessing import cpu_count
from lockfile import LockFile

from argparse import ArgumentParser
from argparse import RawTextHelpFormatter
from nipype import config as ncfg

from mriqc.reports.generators import workflow_report
from mriqc.workflows import core as mwc
from mriqc import __version__


def main():
    """Entry point"""
    parser = ArgumentParser(description='MRI Quality Control',
                            formatter_class=RawTextHelpFormatter)

    g_input = parser.add_argument_group('Inputs')
    g_input.add_argument('-B', '--bids-root', action='store', default=os.getcwd())
    g_input.add_argument('-i', '--input-folder', action='store')
    g_input.add_argument('-S', '--subject-id', nargs='*', action='store')
    g_input.add_argument('-s', '--session-id', action='store')
    g_input.add_argument('-r', '--run-id', action='store')
    g_input.add_argument('-d', '--data-type', action='store', nargs='*',
                         choices=['anat', 'func'], default=['anat', 'func'])
    g_input.add_argument('-v', '--version', action='store_true', default=False,
                         help='Show current mriqc version')

    g_input.add_argument('--nthreads', action='store', default=0,
                         type=int, help='number of threads')
    g_input.add_argument('--write-graph', action='store_true', default=False,
                         help='Write workflow graph.')
    g_input.add_argument('--test-run', action='store_true', default=False,
                         help='Do not run the workflow.')
    g_input.add_argument('--use-plugin', action='store', default=None,
                         help='nipype plugin configuration file')

    g_input.add_argument('--save-memory', action='store_true', default=False,
                         help='Save as much memory as possible')
    g_input.add_argument('--hmc-afni', action='store_true', default=False,
                         help='Use ANFI 3dvolreg for head motion correction (HMC) and '
                              'frame displacement (FD) estimation')
    g_input.add_argument('--ants-settings', action='store',
                         help='path to JSON file with settings for ANTS')


    g_outputs = parser.add_argument_group('Outputs')
    g_outputs.add_argument('-o', '--output-dir', action='store')
    g_outputs.add_argument('-w', '--work-dir', action='store', default=op.join(os.getcwd(), 'work'))

    opts = parser.parse_args()

    bids_root = op.abspath(opts.bids_root)
    if opts.input_folder is not None:
        warn('The --input-folder flag is deprecated, please use -B instead', DeprecationWarning)

        if bids_root == os.getcwd():
            bids_root = op.abspath(opts.input_folder)

    if opts.version:
        print('mriqc version ' + __version__)
        exit(0)

    settings = {'bids_root': bids_root,
                'output_dir': os.getcwd(),
                'write_graph': opts.write_graph,
                'save_memory': opts.save_memory,
                'hmc_afni': opts.hmc_afni,
                'nthreads': opts.nthreads}

    if opts.output_dir:
        settings['output_dir'] = op.abspath(opts.output_dir)

    if not op.exists(settings['output_dir']):
        os.makedirs(settings['output_dir'])

    settings['work_dir'] = op.abspath(opts.work_dir)

    with LockFile(settings['work_dir']):
        if not op.exists(settings['work_dir']):
            os.makedirs(settings['work_dir'])

    if opts.ants_settings:
        settings['ants_settings'] = opts.ants_settings

    log_dir = op.join(settings['work_dir'] + '_log')
    if not op.exists(log_dir):
        os.makedirs(log_dir)

    # Set nipype config
    ncfg.update_config({
        'logging': {'log_directory': log_dir, 'log_to_file': True},
        'execution': {'crashdump_dir': log_dir}
    })

    plugin_settings = {'plugin': 'Linear'}
    if opts.use_plugin is not None:
        from yaml import load as loadyml
        with open(opts.use_plugin) as pfile:
            plugin_settings = loadyml(pfile)
    else:
        # Setup multiprocessing
        if settings['nthreads'] == 0:
            settings['nthreads'] = cpu_count()

        if settings['nthreads'] > 1:
            plugin_settings['plugin'] = 'MultiProc'
            plugin_settings['plugin_args'] = {'n_procs': settings['nthreads']}

    for dtype in opts.data_type:
        ms_func = getattr(mwc, 'ms_' + dtype)
        workflow = ms_func(subject_id=opts.subject_id, session_id=opts.session_id,
                           run_id=opts.run_id, settings=settings)
        workflow.base_dir = settings['work_dir']
        if settings.get('write_graph', False):
            workflow.write_graph()

        if not opts.test_run:
            workflow.run(**plugin_settings)

        if opts.subject_id is None and not opts.test_run:
            workflow_report(dtype, settings)


if __name__ == '__main__':
    main()
