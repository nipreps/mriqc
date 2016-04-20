#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: oesteban
# @Date:   2015-11-19 16:44:27
# @Last Modified by:   oesteban
# @Last Modified time: 2016-04-20 09:35:19

"""
=====
MRIQC
=====
"""
import os
import os.path as op
from multiprocessing import cpu_count

from argparse import ArgumentParser
from argparse import RawTextHelpFormatter
from nipype import config as ncfg

from mriqc.workflows import anat_qc_workflow
from mriqc import __version__


def main():
    """Entry point"""
    parser = ArgumentParser(description='MRI Quality Control',
                            formatter_class=RawTextHelpFormatter)

    g_input = parser.add_argument_group('Inputs')
    g_input.add_argument('-B', '--bids-root', action='store', default=os.getcwd())
    g_input.add_argument('-S', '--subject-id', action='store', required=True)
    g_input.add_argument('-s', '--session-id', action='store', default='single_session')
    g_input.add_argument('-r', '--run-id', action='store', default='single_run')
    g_input.add_argument('-d', '--data-type', action='store', choices=['anat', 'func'])
    g_input.add_argument('-v', '--version', action='store_true', default=False,
                         help='Show current mriqc version')

    g_input.add_argument('--nthreads', action='store', default=0,
                         type=int, help='number of threads')
    g_input.add_argument('--write-graph', action='store_true', default=False,
                         help='Write workflow graph.')
    g_input.add_argument('--use-plugin', action='store', default=None,
                         help='nipype plugin configuration file')

    g_input.add_argument('--save-memory', action='store_true', default=False,
                         help='Save as much memory as possible')


    g_outputs = parser.add_argument_group('Outputs')
    g_outputs.add_argument('-o', '--output-dir', action='store')
    g_outputs.add_argument('-w', '--work-dir', action='store', default=op.join(os.getcwd(), 'work'))

    opts = parser.parse_args()

    if opts.version:
        print 'mriqc version ' + __version__
        exit(0)

    settings = {'bids_root': op.abspath(opts.bids_root),
                'output_dir': os.getcwd(),
                'write_graph': opts.write_graph,
                'save_memory': opts.save_memory,
                'nthreads': opts.nthreads}

    if opts.output_dir:
        settings['output_dir'] = op.abspath(opts.output_dir)

    if not op.exists(settings['output_dir']):
        os.makedirs(settings['output_dir'])

    settings['work_dir'] = op.abspath(opts.work_dir)
    if not op.exists(settings['work_dir']):
        os.makedirs(settings['work_dir'])

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

    settings['formatted_name'] = 'sub-%s' % opts.subject_id
    if opts.session_id is not None:
        settings['formatted_name'] += '_ses-%s' % opts.session_id
    if opts.run_id is not None:
        settings['formatted_name'] += '_run-%s' % opts.run_id

    if opts.data_type == 'anat':
        workflow = anat_qc_workflow(name='mriqc_sub_' + opts.subject_id, settings=settings)

    workflow.inputs.inputnode.bids_root = opts.bids_root
    workflow.inputs.inputnode.subject_id = opts.subject_id
    workflow.inputs.inputnode.session_id = opts.session_id
    workflow.inputs.inputnode.run_id = opts.run_id
    workflow.inputs.inputnode.data_type = opts.data_type
    workflow.run(**plugin_settings)


if __name__ == '__main__':
    main()
