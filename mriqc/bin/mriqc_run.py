#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: oesteban
# @Date:   2015-11-19 16:44:27

"""
=====
MRIQC
=====
"""
from __future__ import print_function, division, absolute_import, unicode_literals

import os
import os.path as op
from multiprocessing import cpu_count

from .. import __version__

DEFAULT_MEM_GB = 8

def get_parser():
    """Build parser object"""
    from argparse import ArgumentParser
    from argparse import RawTextHelpFormatter
    from .. import DEFAULTS

    parser = ArgumentParser(description='MRIQC: MRI Quality Control',
                            formatter_class=RawTextHelpFormatter)

    # Arguments as specified by BIDS-Apps
    # required, positional arguments
    # IMPORTANT: they must go directly with the parser object
    parser.add_argument('bids_dir', action='store',
                        help='The directory with the input dataset '
                             'formatted according to the BIDS standard.')
    parser.add_argument('output_dir', action='store',
                        help='The directory where the output files '
                             'should be stored. If you are running group level analysis '
                             'this folder should be prepopulated with the results of the'
                             'participant level analysis.')
    parser.add_argument('analysis_level', action='store', nargs='+',
                        help='Level of the analysis that will be performed. '
                             'Multiple participant level analyses can be run independently '
                             '(in parallel) using the same output_dir.',
                        choices=['participant', 'group'])

    # optional arguments
    parser.add_argument('--version', action='version',
                        version='mriqc v{}'.format(__version__))

    # BIDS selectors
    g_bids = parser.add_argument_group('Options for filtering BIDS queries')
    g_bids.add_argument('--participant_label', '--participant-label', action='store', nargs='+',
                        help='one or more participant identifiers (the sub- prefix can be '
                             'removed)')
    g_bids.add_argument('--session-id', action='store', nargs='+',
                        help='select a specific session to be processed')
    g_bids.add_argument('--run-id', action='store', type=str, nargs='+',
                        help='select a specific run to be processed')
    g_bids.add_argument('--task-id', action='store', nargs='+', type=str,
                        help='select a specific task to be processed')
    g_bids.add_argument('-m', '--modalities', action='store', nargs='*',
                        choices=['T1w', 'bold', 'T2w'], default=['T1w', 'bold', 'T2w'],
                        help='select one of the supported MRI types')

    # Control instruments
    g_outputs = parser.add_argument_group('Instrumental options')
    g_outputs.add_argument('-w', '--work-dir', action='store',
                           default=op.join(os.getcwd(), 'work'))
    g_outputs.add_argument('--report-dir', action='store')
    g_outputs.add_argument('--verbose-reports', default=False, action='store_true')
    g_outputs.add_argument('--write-graph', action='store_true', default=False,
                           help='Write workflow graph.')
    g_outputs.add_argument('--dry-run', action='store_true', default=False,
                           help='Do not run the workflow.')
    g_outputs.add_argument('--profile', action='store_true', default=False,
                           help='hook up the resource profiler callback to nipype')
    g_outputs.add_argument('--use-plugin', action='store', default=None,
                           help='nipype plugin configuration file')
    g_outputs.add_argument('--no-sub', default=False, action='store_true',
                           help='Turn off submission of anonymized quality metrics '
                                'to MRIQC\'s metrics repository.')
    g_outputs.add_argument('--email', action='store', default='', type=str,
                           help='Email address to include with quality metric submission.')
    g_outputs.add_argument("-v", "--verbose", dest="verbose_count",
                           action="count", default=0,
                           help="increases log verbosity for each occurence, debug level is -vvv")

    g_outputs.add_argument(
        '--webapi-url', action='store', default='https://mriqc.nimh.nih.gov/api/v1', type=str,
        help='IP address where the MRIQC WebAPI is listening')
    g_outputs.add_argument(
        '--webapi-port', action='store', type=int,
        help='port where the MRIQC WebAPI is listening')

    g_outputs.add_argument('--upload-strict', action='store_true', default=False,
                           help='upload will fail if if upload is strict')
    # General performance
    g_perfm = parser.add_argument_group('Options to handle performance')
    g_perfm.add_argument('--n_procs', '--nprocs', '--n_cpus', '--nprocs',
                         action='store', default=0, type=int, help='number of threads')
    g_perfm.add_argument('--mem_gb', action='store', default=0, type=int,
                         help='available total memory')
    g_perfm.add_argument('--testing', action='store_true', default=False,
                         help='use testing settings for a minimal footprint')
    g_perfm.add_argument(
        '-f', '--float32', action='store_true', default=DEFAULTS['float32'],
        help="Cast the input data to float32 if it's represented in higher precision "
             "(saves space and improves perfomance)")

    # Workflow settings
    g_conf = parser.add_argument_group('Workflow configuration')
    g_conf.add_argument('--ica', action='store_true', default=False,
                        help='Run ICA on the raw data and include the components'
                             'in the individual reports (slow but potentially very insightful)')
    g_conf.add_argument('--hmc-afni', action='store_true', default=True,
                        help='Use ANFI 3dvolreg for head motion correction (HMC) - default')
    g_conf.add_argument('--hmc-fsl', action='store_true', default=False,
                        help='Use FSL MCFLIRT instead of AFNI for head motion correction (HMC)')
    g_conf.add_argument('--fft-spikes-detector', action='store_true', default=False,
                        help='Turn on FFT based spike detector (slow).')
    g_conf.add_argument('--fd_thres', action='store', default=0.2,
                        type=float, help='motion threshold for FD computation')

    # ANTs options
    g_ants = parser.add_argument_group('Specific settings for ANTs')
    g_ants.add_argument(
        '--ants-nthreads', action='store', type=int, default=0,
        help='number of threads that will be set in ANTs processes')
    g_ants.add_argument('--ants-settings', action='store',
                        help='path to JSON file with settings for ANTS')

    # AFNI head motion correction settings
    g_afni = parser.add_argument_group('Specific settings for AFNI')
    g_afni.add_argument('--deoblique', action='store_true', default=False,
                        help='Deoblique the functional scans during head motion '
                             'correction preprocessing')
    g_afni.add_argument('--despike', action='store_true', default=False,
                        help='Despike the functional scans during head motion correction '
                             'preprocessing')
    g_afni.add_argument('--start-idx', action='store', type=int,
                        help='Initial volume in functional timeseries that should be '
                             'considered for preprocessing')
    g_afni.add_argument('--stop-idx', action='store', type=int,
                        help='Final volume in functional timeseries that should be '
                             'considered for preprocessing')
    g_afni.add_argument('--correct-slice-timing', action='store_true', default=False,
                        help='Perform slice timing correction')
    return parser


def main():
    """Entry point"""
    from niworkflows.nipype import config as ncfg, logging as nlog
    from niworkflows.nipype.pipeline.engine import Workflow

    from .. import logging
    from ..utils.bids import collect_bids_data
    from ..workflows.core import build_workflow
    from ..utils.misc import check_folder

    # Run parser
    opts = get_parser().parse_args()

    # Retrieve logging level
    log_level = int(max(3 - opts.verbose_count, 0) * 10)
    if opts.verbose_count > 1:
        log_level = int(max(25 - 5 * opts.verbose_count, 1))
    print(log_level)

    logging.getLogger().setLevel(log_level)
    log = logging.getLogger('mriqc.cli')

    # Build settings dict
    bids_dir = op.abspath(opts.bids_dir)

    # Number of processes
    n_procs = opts.n_procs

    settings = {
        'bids_dir': bids_dir,
        'write_graph': opts.write_graph,
        'testing': opts.testing,
        'hmc_afni': opts.hmc_afni,
        'hmc_fsl': opts.hmc_fsl,
        'fft_spikes_detector': opts.fft_spikes_detector,
        'n_procs': n_procs,
        'ants_nthreads': opts.ants_nthreads,
        'output_dir': op.abspath(opts.output_dir),
        'work_dir': op.abspath(opts.work_dir),
        'verbose_reports': opts.verbose_reports or opts.testing,
        'float32': opts.float32,
        'ica': opts.ica,
        'no_sub': opts.no_sub,
        'email': opts.email,
        'fd_thres': opts.fd_thres,
        'webapi_url': opts.webapi_url,
        'webapi_port': opts.webapi_port,
        'upload_strict': opts.upload_strict,
    }

    if opts.hmc_afni:
        settings['deoblique'] = opts.deoblique
        settings['despike'] = opts.despike
        settings['correct_slice_timing'] = opts.correct_slice_timing
        if opts.start_idx:
            settings['start_idx'] = opts.start_idx
        if opts. stop_idx:
            settings['stop_idx'] = opts.stop_idx

    if opts.ants_settings:
        settings['ants_settings'] = opts.ants_settings

    log_dir = op.join(settings['output_dir'], 'logs')

    analysis_levels = opts.analysis_level
    if opts.participant_label is None:
        analysis_levels.append('group')
    analysis_levels = list(set(analysis_levels))
    if len(analysis_levels) > 2:
        raise RuntimeError('Error parsing analysis levels, got "%s"' % ', '.join(analysis_levels))

    settings['report_dir'] = opts.report_dir
    if not settings['report_dir']:
        settings['report_dir'] = op.join(settings['output_dir'], 'reports')

    check_folder(settings['output_dir'])
    if 'participant' in analysis_levels:
        check_folder(settings['work_dir'])

    check_folder(log_dir)
    check_folder(settings['report_dir'])

    # Set nipype config
    ncfg.update_config({
        'logging': {'log_directory': log_dir, 'log_to_file': True},
        'execution': {'crashdump_dir': log_dir, 'crashfile_format': 'txt'},
    })

    # Set nipype logging level
    nlog.getLogger('workflow').setLevel(log_level)
    nlog.getLogger('interface').setLevel(log_level)
    nlog.getLogger('filemanip').setLevel(log_level)

    callback_log_path = None
    plugin_settings = {'plugin': 'Linear'}
    if opts.use_plugin is not None:
        from yaml import load as loadyml
        with open(opts.use_plugin) as pfile:
            plugin_settings = loadyml(pfile)
    else:
        # Setup multiprocessing
        if settings['n_procs'] == 0:
            settings['n_procs'] = cpu_count()

        if settings['ants_nthreads'] == 0:
            if settings['n_procs'] > 1:
                # always leave one extra thread for non ANTs work,
                # don't use more than 8 threads - the speed up is minimal
                settings['ants_nthreads'] = min(settings['n_procs'] - 1, 8)
            else:
                settings['ants_nthreads'] = 1

        if settings['n_procs'] > 1:
            plugin_settings['plugin'] = 'MultiProc'
            plugin_settings['plugin_args'] = {'n_procs': settings['n_procs']}
            if opts.mem_gb:
                plugin_settings['plugin_args']['memory_gb'] = opts.mem_gb

    # Process data types
    modalities = opts.modalities

    dataset = collect_bids_data(
        settings['bids_dir'],
        modalities=modalities,
        participant_label=opts.participant_label,
        session=opts.session_id,
        run=opts.run_id,
        task=opts.task_id,
    )

    # Set up participant level
    if 'participant' in analysis_levels:
        log.info('Participant level started...')
        log.info(
            'Running MRIQC-%s (analysis_levels=[%s], participant_label=%s)\n\tSettings=%s',
            __version__, ', '.join(analysis_levels), opts.participant_label, settings)

        workflow = Workflow(name='workflow_enumerator')
        workflow.base_dir = settings['work_dir']

        wf_list = []
        for mod in modalities:
            if not dataset[mod]:
                log.warning('No %s scans were found in %s', mod, settings['bids_dir'])
                continue

            wf_list.append(build_workflow(dataset[mod], mod, settings=settings))

        if wf_list:
            workflow.add_nodes(wf_list)

            if not opts.dry_run:
                if plugin_settings['plugin'] == 'MultiProc' and opts.profile:
                    import logging
                    from niworkflows.nipype.pipeline.plugins.callback_log import log_nodes_cb
                    plugin_settings['plugin_args']['status_callback'] = log_nodes_cb
                    callback_log_path = op.join(log_dir, 'run_stats.log')
                    logger = logging.getLogger('callback')
                    logger.setLevel(logging.DEBUG)
                    handler = logging.FileHandler(callback_log_path)
                    logger.addHandler(handler)

                # Warn about submitting measures BEFORE
                if not settings['no_sub']:
                    log.warning(
                        'Anonymized quality metrics will be submitted'
                        ' to MRIQC\'s metrics repository.'
                        ' Use --no-sub to disable submission.')

                # run MRIQC
                workflow.run(**plugin_settings)

                # Warn about submitting measures AFTER
                if not settings['no_sub']:
                    log.warning(
                        'Anonymized quality metrics have beeen submitted'
                        ' to MRIQC\'s metrics repository.'
                        ' Use --no-sub to disable submission.')

                if callback_log_path is not None:
                    from niworkflows.nipype.utils.draw_gantt_chart import generate_gantt_chart
                    generate_gantt_chart(callback_log_path, cores=settings['n_procs'])
        else:
            msg = """\
Error reading BIDS directory ({}), or the dataset is not \
BIDS-compliant."""
            if opts.participant_label is not None:
                msg = """\
None of the supplied labels (--participant_label) matched with the \
participants found in the BIDS directory ({})."""
            raise RuntimeError(msg.format(settings['bids_dir']))

        log.info('Participant level finished successfully.')

    # Set up group level
    if 'group' in analysis_levels:
        from ..reports import group_html
        from ..utils.misc import generate_csv  # , generate_pred

        log.info('Group level started...')
        log.info(
            'Running MRIQC-%s (analysis_levels=[%s], participant_label=%s)\n\tSettings=%s',
            __version__, ', '.join(analysis_levels), opts.participant_label, settings)

        reports_dir = check_folder(op.join(settings['output_dir'], 'reports'))
        derivatives_dir = op.join(settings['output_dir'], 'derivatives')

        n_group_reports = 0
        for mod in modalities:
            dataframe, out_csv = generate_csv(derivatives_dir,
                                              settings['output_dir'], mod)

            # If there are no iqm.json files, nothing to do.
            if dataframe is None:
                log.warning(
                    'No IQM-JSON files were found for the %s data type in %s. The group-level '
                    'report was not generated.', mod, derivatives_dir)
                continue

            log.info('Summary CSV table for the %s data generated (%s)', mod, out_csv)

            # out_pred = generate_pred(derivatives_dir, settings['output_dir'], mod)
            # if out_pred is not None:
            #     log.info('Predicted QA CSV table for the %s data generated (%s)',
            #                    mod, out_pred)

            out_html = op.join(reports_dir, mod + '_group.html')
            group_html(out_csv, mod,
                       csv_failed=op.join(settings['output_dir'], 'failed_' + mod + '.csv'),
                       out_file=out_html)
            log.info('Group-%s report generated (%s)', mod, out_html)
            n_group_reports += 1

        if n_group_reports == 0:
            raise Exception("No data found. No group level reports were generated.")

        log.info('Group level finished successfully.')



if __name__ == '__main__':
    main()
