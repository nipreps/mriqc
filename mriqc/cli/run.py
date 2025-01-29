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
"""Definition of the command line interface's (CLI) entry point."""


def format_elapsed_time(elapsed_timedelta):
    """Format a timedelta instance as a %Hh %Mmin %Ss string."""
    return (
        f'{elapsed_timedelta.days * 24 + elapsed_timedelta.seconds // 3600:02d}h '
        f'{(elapsed_timedelta.seconds % 3600) // 60:02d}min '
        f'{elapsed_timedelta.seconds % 60:02d}s'
    )


def main(argv=None):
    """Entry point for MRIQC's CLI."""
    import atexit
    import gc
    import os
    import sys
    import time
    import datetime
    from tempfile import mkstemp

    from mriqc import config, messages
    from mriqc.cli.parser import parse_args

    atexit.register(config.restore_env)

    config.settings.start_time = time.time()

    # Run parser
    parse_args(argv)

    if config.execution.pdb:
        from mriqc.utils.debug import setup_exceptionhook

        setup_exceptionhook()
        config.nipype.plugin = 'Linear'

    # CRITICAL Save the config to a file. This is necessary because the execution graph
    # is built as a separate process to keep the memory footprint low. The most
    # straightforward way to communicate with the child process is via the filesystem.
    # The config file name needs to be unique, otherwise multiple mriqc instances
    # will create write conflicts.
    config_file = config.to_filename()
    config.loggers.cli.info(f'MRIQC config file: {config_file}.')

    exitcode = 0
    # Set up participant level
    if 'participant' in config.workflow.analysis_level:
        _pool = None
        if config.nipype.plugin in ('MultiProc', 'LegacyMultiProc'):
            import multiprocessing as mp
            import multiprocessing.forkserver
            from concurrent.futures import ProcessPoolExecutor
            from contextlib import suppress

            os.environ['OMP_NUM_THREADS'] = '1'
            os.environ['NUMEXPR_MAX_THREADS'] = '1'

            with suppress(RuntimeError):
                mp.set_start_method('fork')
            gc.collect()

            _pool = ProcessPoolExecutor(
                max_workers=config.nipype.nprocs,
                initializer=config._process_initializer,
                initargs=(config_file,),
            )

        _resmon = None
        if config.execution.resource_monitor:
            from mriqc.instrumentation.resources import ResourceRecorder

            _resmon = ResourceRecorder(
                pid=os.getpid(),
                log_file=mkstemp(
                    dir=config.execution.work_dir, prefix='.resources.', suffix='.tsv'
                )[1],
            )
            _resmon.start()

        if not config.execution.notrack:
            from ..utils.telemetry import setup_migas

            setup_migas()

        # CRITICAL Call build_workflow(config_file, retval) in a subprocess.
        # Because Python on Linux does not ever free virtual memory (VM), running the
        # workflow construction jailed within a process preempts excessive VM buildup.
        from multiprocessing import Manager, Process

        with Manager() as mgr:
            from .workflow import build_workflow

            retval = mgr.dict()
            p = Process(target=build_workflow, args=(str(config_file), retval))
            p.start()
            p.join()

            mriqc_wf = retval.get('workflow', None)
            exitcode = p.exitcode or retval.get('return_code', 0)

        # CRITICAL Load the config from the file. This is necessary because the ``build_workflow``
        # function executed constrained in a process may change the config (and thus the global
        # state of MRIQC).
        config.load(config_file)

        exitcode = exitcode or (mriqc_wf is None) * os.EX_SOFTWARE
        if exitcode != 0:
            sys.exit(exitcode)

        # Initialize nipype config
        config.nipype.init()
        # Make sure loggers are started
        config.loggers.init()

        if _resmon:
            config.loggers.cli.info(f'Started resource recording at {_resmon._logfile}.')

        # Resource management options
        if config.nipype.plugin in ('MultiProc', 'LegacyMultiProc') and (
            1 < config.nipype.nprocs < config.nipype.omp_nthreads
        ):
            config.loggers.cli.warning(
                'Per-process threads (--omp-nthreads=%d) exceed total '
                'threads (--nthreads/--n_cpus=%d)',
                config.nipype.omp_nthreads,
                config.nipype.nprocs,
            )

        # Check synthstrip is properly installed
        if not config.environment.synthstrip_path:
            config.loggers.cli.warning(
                (
                    'Please make sure FreeSurfer is installed and the FREESURFER_HOME '
                    'environment variable is defined and pointing at the right directory.'
                )
                if config.environment.freesurfer_home is None
                else (
                    f'FreeSurfer seems to be installed at {config.environment.freesurfer_home},'
                    " however SynthStrip's model is not found at the expected path."
                )
            )

        if mriqc_wf is None:
            sys.exit(os.EX_SOFTWARE)

        if mriqc_wf and config.execution.write_graph:
            mriqc_wf.write_graph(graph2use='colored', format='svg', simple_form=True)

        if not config.execution.dry_run and not config.execution.reports_only:
            # Warn about submitting measures BEFORE
            if not config.execution.no_sub:
                config.loggers.cli.warning(config.DSA_MESSAGE)

            # Clean up master process before running workflow, which may create forks
            gc.collect()
            # run MRIQC
            _plugin = config.nipype.get_plugin()
            if _pool:
                from mriqc.engine.plugin import MultiProcPlugin

                _plugin = {
                    'plugin': MultiProcPlugin(pool=_pool, plugin_args=config.nipype.plugin_args),
                }
            mriqc_wf.run(**_plugin)

            # Warn about submitting measures AFTER
            if not config.execution.no_sub:
                config.loggers.cli.warning(config.DSA_MESSAGE)

        if not config.execution.dry_run:
            from mriqc.reports.individual import generate_reports

            generate_reports()

        _subject_duration = (time.time() - config.settings.start_time) / sum(
            len(files) for files in config.workflow.inputs.values()
        )
        _subject_duration_td = datetime.timedelta(seconds=_subject_duration)
        time_strf = format_elapsed_time(_subject_duration_td)

        config.loggers.cli.log(
            25,
            messages.PARTICIPANT_FINISHED.format(duration=time_strf),
        )

        if _resmon is not None:
            from mriqc.instrumentation.viz import plot

            _resmon.stop()
            plot(
                _resmon._logfile,
                param='mem_rss_mb',
                out_file=str(_resmon._logfile).replace('.tsv', '.rss.png'),
            )
            plot(
                _resmon._logfile,
                param='mem_vsm_mb',
                out_file=str(_resmon._logfile).replace('.tsv', '.vsm.png'),
            )

    # Set up group level
    if 'group' in config.workflow.analysis_level:
        from mriqc.reports.group import gen_html as group_html

        from ..utils.misc import generate_tsv  # , generate_pred

        config.loggers.cli.log(26, messages.GROUP_START)

        # Generate reports
        mod_group_reports = []
        for mod in config.execution.modalities or config.SUPPORTED_SUFFIXES:
            output_dir = config.execution.output_dir
            dataframe, out_tsv = generate_tsv(output_dir, mod)
            # If there are no iqm.json files, nothing to do.
            if dataframe is None:
                continue

            tsv_message = messages.TSV_GENERATED.format(modality=mod, path=out_tsv)
            config.loggers.cli.info(tsv_message)

            # out_pred = generate_pred(derivatives_dir, settings['output_dir'], mod)
            # if out_pred is not None:
            #     log.info('Predicted QA CSV table for the %s data generated (%s)',
            #                    mod, out_pred)

            out_html = output_dir / f'group_{mod}.html'
            group_html(
                out_tsv,
                mod,
                csv_failed=output_dir / f'group_variant-failed_{mod}.csv',
                out_file=out_html,
            )
            report_message = messages.GROUP_REPORT_GENERATED.format(modality=mod, path=out_html)
            config.loggers.cli.info(report_message)
            mod_group_reports.append(mod)

        if not mod_group_reports:
            raise Exception(messages.GROUP_NO_DATA)

        config.loggers.cli.info(messages.GROUP_FINISHED)

    from mriqc.utils.bids import write_bidsignore, write_derivative_description

    config.loggers.cli.info(messages.BIDS_META)
    write_derivative_description(config.execution.bids_dir, config.execution.output_dir)
    write_bidsignore(config.execution.output_dir)

    _run_duration = time.time() - config.settings.start_time
    _run_duration_td = datetime.timedelta(seconds=_run_duration)
    time_strf = format_elapsed_time(_run_duration_td)

    config.loggers.cli.log(
        26,
        messages.RUN_FINISHED.format(duration=time_strf),
    )
    config.to_filename(
        config.execution.log_dir / f'config-{config.execution.run_uuid}.toml',
        store_inputs=False,  # Inputs are not necessary anymore
    )
    sys.exit(exitcode)


if __name__ == '__main__':
    main()
