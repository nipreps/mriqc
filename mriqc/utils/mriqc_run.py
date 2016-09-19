#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: oesteban
# @Date:   2015-11-19 16:44:27
# @Last Modified by:   oesteban
# @Last Modified time: 2016-09-16 11:47:50

"""
=====
MRIQC
=====
"""
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from __future__ import unicode_literals

import os
import os.path as op
from multiprocessing import cpu_count
from nipype import logging, config as ncfg
from lockfile import LockFile

from argparse import ArgumentParser
from argparse import RawTextHelpFormatter

from mriqc.workflows import core as mwc
from mriqc import __version__

LOGGER = logging.getLogger('workflow')


def main():
    """Entry point"""
    parser = ArgumentParser(description='MRI Quality Control',
                            formatter_class=RawTextHelpFormatter)

    parser.add_argument('-v', '--version', action='version',
                        version='mriqc v{}'.format(__version__))

    parser.add_argument('bids_dir', action='store',
                        help='The directory with the input dataset '
                             'formatted according to the BIDS standard.')
    parser.add_argument('output_dir', action='store',
                        help='The directory where the output files '
                             'should be stored. If you are running group level analysis '
                             'this folder should be prepopulated with the results of the'
                             'participant level analysis.')
    parser.add_argument('analysis_level', action='store',
                        help='Level of the analysis that will be performed. '
                             'Multiple participant level analyses can be run independently '
                             '(in parallel) using the same output_dir.',
                        choices=['participant', 'group'])
    parser.add_argument('--participant_label', '--subject_list', '-S', action='store',
                        help='The label(s) of the participant(s) that should be analyzed. '
                             'The label corresponds to sub-<participant_label> from the '
                             'BIDS spec (so it does not include "sub-"). If this parameter '
                             'is not provided all subjects should be analyzed. Multiple '
                             'participants can be specified with a space separated list.',
                        nargs="*")

    g_input = parser.add_argument_group('mriqc specific inputs')
    g_input.add_argument('-d', '--data-type', action='store', nargs='*',
                         choices=['anat', 'func'], default=['anat', 'func'])
    g_input.add_argument('-s', '--session-id', action='store')
    g_input.add_argument('-r', '--run-id', action='store')
    g_input.add_argument('--nthreads', action='store', default=0,
                         type=int, help='number of threads')
    g_input.add_argument('--write-graph', action='store_true', default=False,
                         help='Write workflow graph.')
    g_input.add_argument('--dry-run', action='store_true', default=False,
                         help='Do not run the workflow.')
    g_input.add_argument('--use-plugin', action='store', default=None,
                         help='nipype plugin configuration file')

    g_input.add_argument('--testing', action='store_true', default=False,
                         help='use testing settings for a minimal footprint')
    g_input.add_argument('--hmc-afni', action='store_true', default=False,
                         help='Use ANFI 3dvolreg for head motion correction (HMC) and '
                              'frame displacement (FD) estimation')


    g_outputs = parser.add_argument_group('mriqc specific outputs')
    g_outputs.add_argument('-w', '--work-dir', action='store', default=op.join(os.getcwd(), 'work'))

    # ANTs options
    g_ants = parser.add_argument_group('specific settings for ANTs registrations')
    g_ants.add_argument('--ants-nthreads', action='store', type=int,
                        help='number of threads that will be set in ANTs processes')
    g_ants.add_argument('--ants-settings', action='store',
                         help='path to JSON file with settings for ANTS')

    opts = parser.parse_args()


    # Build settings dict
    bids_dir = op.abspath(opts.bids_dir)
    settings = {
        'bids_dir': bids_dir,
        'write_graph': opts.write_graph,
        'testing': opts.testing,
        'hmc_afni': opts.hmc_afni,
        'nthreads': opts.nthreads,
        'output_dir': op.abspath(opts.output_dir),
        'work_dir': op.abspath(opts.work_dir)
    }

    if opts.ants_settings:
        settings['ants_settings'] = opts.ants_settings

    if opts.ants_nthreads:
        settings['ants_nthreads'] = opts.ants_nthreads

    log_dir = op.join(settings['output_dir'], 'logs')

    with LockFile('.mriqc-lock'):
        if not op.exists(settings['output_dir']):
            os.makedirs(settings['output_dir'])

        if not op.exists(settings['work_dir']):
            os.makedirs(settings['work_dir'])

        if not op.exists(log_dir):
            os.makedirs(log_dir)

        if not op.exists(op.join(settings['work_dir'], 'reports')):
            os.makedirs(op.join(settings['work_dir'], 'reports'))

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

    LOGGER.info("""Running MRIQC version %s:
            analysis_level=%s
            participant_label=%s
            settings=%s""", __version__, opts.analysis_level, opts.participant_label, settings)

    # Set up participant level
    if opts.analysis_level == 'participant':
        for dtype in opts.data_type:
            ms_func = getattr(mwc, 'ms_' + dtype)
            workflow = ms_func(subject_id=opts.participant_label, session_id=opts.session_id,
                               run_id=opts.run_id, settings=settings)
            if workflow is None:
                LOGGER.warn('No scans were found for the given inputs')
                continue

            workflow.base_dir = settings['work_dir']
            if settings.get('write_graph', False):
                workflow.write_graph()

            if not opts.dry_run:
                workflow.run(**plugin_settings)

    # Set up group level
    elif opts.analysis_level == 'group':
        from mriqc.reports import MRIQCReportPDF

        for dtype in opts.data_type:
            reporter = MRIQCReportPDF(dtype, settings)
            reporter.group_report()
            reporter.individual_report()



if __name__ == '__main__':
    main()
