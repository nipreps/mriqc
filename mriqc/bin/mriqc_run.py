#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: oesteban
# @Date:   2015-11-19 16:44:27

"""
=====
MRIQC
=====
"""
from os import cpu_count
import logging
import gc
from pathlib import Path
import matplotlib

DSA_MESSAGE = """\
Anonymized quality metrics (IQMs) will be submitted to MRIQC's metrics repository. \
Submission of IQMs can be disabled using the ``--no-sub`` argument. \
Please visit https://mriqc.readthedocs.io/en/latest/dsa.html to revise MRIQC's \
Data Sharing Agreement."""

matplotlib.use('Agg')  # Replace matplotlib's backend ASAP (see #758)
logging.addLevelName(25, 'IMPORTANT')  # Add a new level between INFO and WARNING
logging.addLevelName(15, 'VERBOSE')  # Add a new level between INFO and DEBUG
logging.captureWarnings(True)
DEFAULT_MEM_GB = 8


def get_parser():
    """Build parser object"""
    from argparse import ArgumentParser
    from argparse import RawTextHelpFormatter
    from .. import DEFAULTS, __description__, __version__

    parser = ArgumentParser(
        description="""MRIQC: MRI Quality Control\n\n\
%s
%s""" % (__description__, DSA_MESSAGE),
        formatter_class=RawTextHelpFormatter)

    # Arguments as specified by BIDS-Apps
    # required, positional arguments
    # IMPORTANT: they must go directly with the parser object
    parser.add_argument('bids_dir', action='store', type=Path,
                        help='The directory with the input dataset '
                             'formatted according to the BIDS standard.')
    parser.add_argument('output_dir', action='store', type=Path,
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
    g_bids = parser.add_argument_group('Options for filtering the input BIDS dataset')
    g_bids.add_argument('--participant_label', '--participant-label', action='store', nargs='*',
                        help='one or more participant identifiers (the sub- prefix can be '
                             'removed)')
    g_bids.add_argument('--session-id', action='store', nargs='*', type=str,
                        help='filter input dataset by session id')
    g_bids.add_argument('--run-id', action='store', type=int, nargs='*',
                        help='filter input dataset by run id '
                             '(only integer run ids are valid)')
    g_bids.add_argument('--task-id', action='store', nargs='*', type=str,
                        help='filter input dataset by task id')
    g_bids.add_argument('-m', '--modalities', action='store', nargs='*',
                        help='filter input dataset by MRI type')
    g_bids.add_argument('--dsname', type=str, help='a dataset name')

    # Control instruments
    g_outputs = parser.add_argument_group('Instrumental options')
    g_outputs.add_argument('-w', '--work-dir', action='store', default=Path() / 'work',
                           type=Path, help='change the folder to store intermediate results')
    g_outputs.add_argument('--verbose-reports', default=False, action='store_true')
    g_outputs.add_argument('--write-graph', action='store_true', default=False,
                           help='Write workflow graph.')
    g_outputs.add_argument('--dry-run', action='store_true', default=False,
                           help='Do not run the workflow.')
    g_outputs.add_argument('--profile', action='store_true', default=False,
                           help='hook up the resource profiler callback to nipype')
    g_outputs.add_argument('--use-plugin', action='store', default=None, type=Path,
                           help='nipype plugin configuration file')
    g_outputs.add_argument('--no-sub', default=False, action='store_true',
                           help='Turn off submission of anonymized quality metrics '
                                'to MRIQC\'s metrics repository.')
    g_outputs.add_argument('--email', action='store', default='', type=str,
                           help='Email address to include with quality metric submission.')
    g_outputs.add_argument("-v", "--verbose", dest="verbose_count", action="count", default=0,
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
        '--ants-nthreads', action='store', type=int, default=1,
        help='number of threads that will be set in ANTs processes')
    g_ants.add_argument(
        '--ants-float', action='store_true', default=False,
        help='use float number precision on ANTs computations')
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
    import sys
    from nipype import logging as nlogging
    from multiprocessing import set_start_method, Process, Manager
    from .. import __version__

    set_start_method('forkserver')

    # Run parser
    opts = get_parser().parse_args()

    # Analysis levels
    analysis_levels = set(opts.analysis_level)
    if not opts.participant_label:
        analysis_levels.add('group')

    # Retrieve logging level
    log_level = int(max(25 - 5 * opts.verbose_count, 1))

    # Set logging level
    logging.getLogger('mriqc').setLevel(log_level)
    nlogging.getLogger('nipype.workflow').setLevel(log_level)
    nlogging.getLogger('nipype.interface').setLevel(log_level)
    nlogging.getLogger('nipype.utils').setLevel(log_level)

    logger = logging.getLogger('mriqc')
    INIT_MSG = """
    Running MRIQC version {version}:
      * BIDS dataset path: {bids_dir}.
      * Output folder: {output_dir}.
      * Analysis levels: {levels}.
    """.format(
        version=__version__,
        bids_dir=opts.bids_dir.expanduser().resolve(),
        output_dir=opts.output_dir.expanduser().resolve(),
        levels=', '.join(reversed(list(analysis_levels)))
    )
    logger.log(25, INIT_MSG)

    # Set up participant level
    if 'participant' in analysis_levels:
        logger.info('Participant level started. Checking BIDS dataset...')

        # Call build_workflow(opts, retval)
        with Manager() as mgr:
            retval = mgr.dict()
            p = Process(target=init_mriqc, args=(opts, retval))
            p.start()
            p.join()

            if p.exitcode != 0:
                sys.exit(p.exitcode)

            mriqc_wf = retval['workflow']
            plugin_settings = retval['plugin_settings']
            subject_list = retval['subject_list']

        if not subject_list:
            logger.critical(
                'MRIQC did not find any target image file under the given BIDS '
                'folder (%s). Please check that the dataset is BIDS valid at '
                'http://bids-standard.github.io/bids-validator/ .', opts.bids_dir.resolve())

            bids_selectors = []
            for entity in ['participant-label', 'modalities', 'session-id', 'task-id', 'run-id']:
                values = getattr(opts, entity.replace('-', '_'), None)
                if values:
                    bids_selectors += ['--%s %s' % (entity, ' '.join(values))]
            if bids_selectors:
                logger.warning(
                    'The following BIDS entities were selected as filters: %s. '
                    'Please, check whether their combinations are possible.',
                    ', '.join(bids_selectors)
                )
            sys.exit(1)

        if mriqc_wf is None:
            logger.error('Failed to create the MRIQC workflow, please report the issue '
                         'to https://github.com/poldracklab/mriqc/issues')
            sys.exit(1)

        # Clean up master process before running workflow, which may create forks
        gc.collect()
        if not opts.dry_run:
            # Warn about submitting measures BEFORE
            if not opts.no_sub:
                logger.warning(DSA_MESSAGE)

            # run MRIQC
            mriqc_wf.run(**plugin_settings)

            # Warn about submitting measures AFTER
            if not opts.no_sub:
                logger.warning(DSA_MESSAGE)
        logger.info('Participant level finished successfully.')

    # Set up group level
    if 'group' in analysis_levels:
        from ..utils.bids import DEFAULT_TYPES
        from ..reports import group_html
        from ..utils.misc import generate_tsv  # , generate_pred

        logger.info('Group level started...')

        # Generate reports
        mod_group_reports = []
        for mod in opts.modalities or DEFAULT_TYPES:
            dataframe, out_tsv = generate_tsv(
                opts.output_dir.expanduser().resolve(), mod)
            # If there are no iqm.json files, nothing to do.
            if dataframe is None:
                continue

            logger.info('Generated summary TSV table for the %s data (%s)', mod, out_tsv)

            # out_pred = generate_pred(derivatives_dir, settings['output_dir'], mod)
            # if out_pred is not None:
            #     log.info('Predicted QA CSV table for the %s data generated (%s)',
            #                    mod, out_pred)

            out_html = opts.output_dir / ('group_%s.html' % mod)
            group_html(out_tsv, mod,
                       csv_failed=opts.output_dir / ('group_variant-failed_%s.csv' % mod),
                       out_file=out_html)
            logger.info('Group-%s report generated (%s)', mod, out_html)
            mod_group_reports.append(mod)

        if not mod_group_reports:
            raise Exception("No data found. No group level reports were generated.")

        logger.info('Group level finished successfully.')


def init_mriqc(opts, retval):
    """Build the workflow enumerator"""
    from bids.layout import BIDSLayout
    from nipype import config as ncfg
    from nipype.pipeline.engine import Workflow

    from ..utils.bids import collect_bids_data
    from ..workflows.core import build_workflow

    retval['workflow'] = None
    retval['plugin_settings'] = None

    # Build settings dict
    bids_dir = Path(opts.bids_dir).expanduser()
    output_dir = Path(opts.output_dir).expanduser()

    # Number of processes
    n_procs = opts.n_procs or cpu_count()

    settings = {
        'bids_dir': bids_dir.resolve(),
        'output_dir': output_dir.resolve(),
        'work_dir': opts.work_dir.expanduser().resolve(),
        'write_graph': opts.write_graph,
        'n_procs': n_procs,
        'testing': opts.testing,
        'hmc_afni': opts.hmc_afni,
        'hmc_fsl': opts.hmc_fsl,
        'fft_spikes_detector': opts.fft_spikes_detector,
        'ants_nthreads': opts.ants_nthreads,
        'ants_float': opts.ants_float,
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

    if opts.dsname:
        settings['dataset_name'] = opts.dsname

    log_dir = settings['output_dir'] / 'logs'

    # Create directories
    log_dir.mkdir(parents=True, exist_ok=True)
    settings['work_dir'].mkdir(parents=True, exist_ok=True)

    # Set nipype config
    ncfg.update_config({
        'logging': {'log_directory': str(log_dir), 'log_to_file': True},
        'execution': {
            'crashdump_dir': str(log_dir), 'crashfile_format': 'txt',
            'resource_monitor': opts.profile},
    })

    # Plugin configuration
    plugin_settings = {}
    if n_procs == 1:
        plugin_settings['plugin'] = 'Linear'

        if settings['ants_nthreads'] == 0:
            settings['ants_nthreads'] = 1
    else:
        plugin_settings['plugin'] = 'MultiProc'
        plugin_settings['plugin_args'] = {
            'n_procs': n_procs,
            'raise_insufficient': False,
            'maxtasksperchild': 1,
        }
        if opts.mem_gb:
            plugin_settings['plugin_args']['memory_gb'] = opts.mem_gb

        if settings['ants_nthreads'] == 0:
            # always leave one extra thread for non ANTs work,
            # don't use more than 8 threads - the speed up is minimal
            settings['ants_nthreads'] = min(settings['n_procs'] - 1, 8)

    # Overwrite options if --use-plugin provided
    if opts.use_plugin and opts.use_plugin.exists():
        from yaml import load as loadyml
        with opts.use_plugin.open() as pfile:
            plugin_settings.update(loadyml(pfile))

    layout = BIDSLayout(str(settings['bids_dir']),
                        exclude=['derivatives', 'sourcedata', r'^\..*'])
    dataset = collect_bids_data(
        layout,
        participant_label=opts.participant_label,
        session=opts.session_id,
        run=opts.run_id,
        task=opts.task_id,
        bids_type=opts.modalities,
    )

    workflow = Workflow(name='workflow_enumerator')
    workflow.base_dir = settings['work_dir']
    modalities = [mod for mod, val in dataset.items() if val]

    wf_list = []
    subject_list = []
    for mod in modalities:
        if dataset[mod]:
            wf_list.append(build_workflow(dataset[mod], mod, settings=settings))
            subject_list += dataset[mod]

    retval['subject_list'] = subject_list
    if not wf_list:
        retval['return_code'] = 1
        return retval

    workflow.add_nodes(wf_list)
    retval['plugin_settings'] = plugin_settings
    retval['workflow'] = workflow
    retval['return_code'] = 0
    return retval


if __name__ == '__main__':
    main()
