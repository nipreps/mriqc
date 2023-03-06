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
EXITCODE: int = -1


def main():
    """Entry point for MRIQC's CLI."""
    import gc
    import os
    import sys
    from tempfile import mktemp
    import atexit
    from mriqc import config, messages
    from mriqc.cli.parser import parse_args

    atexit.register(config.restore_env)

    # Run parser
    parse_args()

    if config.execution.pdb:
        from mriqc.utils.debug import setup_exceptionhook

        setup_exceptionhook()
        config.nipype.plugin = "Linear"

    # CRITICAL Save the config to a file. This is necessary because the execution graph
    # is built as a separate process to keep the memory footprint low. The most
    # straightforward way to communicate with the child process is via the filesystem.
    # The config file name needs to be unique, otherwise multiple mriqc instances
    # will create write conflicts.
    config_file = mktemp(
        dir=config.execution.work_dir, prefix=".mriqc.", suffix=".toml"
    )
    config.to_filename(config_file)

    # Set up participant level
    if "participant" in config.workflow.analysis_level:
        _pool = None
        if config.nipype.plugin in ("MultiProc", "LegacyMultiProc"):
            from contextlib import suppress
            import multiprocessing as mp
            import multiprocessing.forkserver
            from concurrent.futures import ProcessPoolExecutor

            os.environ["OMP_NUM_THREADS"] = "1"

            with suppress(RuntimeError):
                mp.set_start_method("fork")
            gc.collect()

            _pool = ProcessPoolExecutor(
                max_workers=config.nipype.nprocs,
                initializer=config._process_initializer,
                initargs=(config.execution.cwd, config.nipype.omp_nthreads),
            )

        _resmon = None
        if config.execution.resource_monitor:
            from mriqc.instrumentation.resources import ResourceRecorder

            _resmon = ResourceRecorder(
                pid=os.getpid(),
                log_file=mktemp(
                    dir=config.execution.work_dir, prefix=".resources.", suffix=".tsv"
                ),
            )
            _resmon.start()

        if not config.execution.notrack:
            from ..utils.telemetry import setup_migas

            setup_migas(init=True)
            atexit.register(migas_exit)

        # CRITICAL Call build_workflow(config_file, retval) in a subprocess.
        # Because Python on Linux does not ever free virtual memory (VM), running the
        # workflow construction jailed within a process preempts excessive VM buildup.
        from multiprocessing import Manager, Process

        global EXITCODE
        with Manager() as mgr:
            from .workflow import build_workflow

            retval = mgr.dict()
            p = Process(target=build_workflow, args=(str(config_file), retval))
            p.start()
            p.join()

            mriqc_wf = retval.get("workflow", None)
            EXITCODE = p.exitcode or retval.get("return_code", 0)

        # CRITICAL Load the config from the file. This is necessary because the ``build_workflow``
        # function executed constrained in a process may change the config (and thus the global
        # state of MRIQC).
        config.load(config_file)

        EXITCODE = EXITCODE or (mriqc_wf is None) * os.EX_SOFTWARE
        if EXITCODE != 0:
            sys.exit(EXITCODE)

        # Initialize nipype config
        config.nipype.init()
        # Make sure loggers are started
        config.loggers.init()

        if _resmon:
            config.loggers.cli.info(
                f"Started resource recording at {_resmon._logfile}."
            )

        # Resource management options
        if config.nipype.plugin in ("MultiProc", "LegacyMultiProc") and (
            1 < config.nipype.nprocs < config.nipype.omp_nthreads
        ):
            config.loggers.cli.warning(
                "Per-process threads (--omp-nthreads=%d) exceed total "
                "threads (--nthreads/--n_cpus=%d)",
                config.nipype.omp_nthreads,
                config.nipype.nprocs,
            )

        if mriqc_wf is None:
            sys.exit(os.EX_SOFTWARE)

        if mriqc_wf and config.execution.write_graph:
            mriqc_wf.write_graph(graph2use="colored", format="svg", simple_form=True)

        if not config.execution.dry_run:
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
                    "plugin": MultiProcPlugin(
                        pool=_pool, plugin_args=config.nipype.plugin_args
                    ),
                }
            mriqc_wf.run(**_plugin)

            # Warn about submitting measures AFTER
            if not config.execution.no_sub:
                config.loggers.cli.warning(config.DSA_MESSAGE)
        config.loggers.cli.log(25, messages.PARTICIPANT_FINISHED)

        if _resmon is not None:
            from mriqc.instrumentation.viz import plot
            _resmon.stop()
            plot(
                _resmon._logfile,
                param="mem_rss_mb",
                out_file=str(_resmon._logfile).replace(".tsv", ".rss.png"),
            )
            plot(
                _resmon._logfile,
                param="mem_vsm_mb",
                out_file=str(_resmon._logfile).replace(".tsv", ".vsm.png"),
            )

    # Set up group level
    if "group" in config.workflow.analysis_level:
        from ..reports import group_html
        from ..utils.bids import DEFAULT_TYPES
        from ..utils.misc import generate_tsv  # , generate_pred

        config.loggers.cli.info(messages.GROUP_START)

        # Generate reports
        mod_group_reports = []
        for mod in config.execution.modalities or DEFAULT_TYPES:
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

            out_html = output_dir / f"group_{mod}.html"
            group_html(
                out_tsv,
                mod,
                csv_failed=output_dir / f"group_variant-failed_{mod}.csv",
                out_file=out_html,
            )
            report_message = messages.GROUP_REPORT_GENERATED.format(
                modality=mod, path=out_html
            )
            config.loggers.cli.info(report_message)
            mod_group_reports.append(mod)

        if not mod_group_reports:
            raise Exception(messages.GROUP_NO_DATA)

        config.loggers.cli.info(messages.GROUP_FINISHED)

    from mriqc.utils.bids import write_bidsignore, write_derivative_description

    config.loggers.cli.info(messages.BIDS_META)
    write_derivative_description(config.execution.bids_dir, config.execution.output_dir)
    write_bidsignore(config.execution.output_dir)
    config.loggers.cli.info(messages.RUN_FINISHED)


def migas_exit() -> None:
    """
    Send a final crumb to the migas server signaling if the run successfully completed
    This function should be registered with `atexit` to run at termination.
    """
    import sys
    from ..utils.telemetry import send_breadcrumb

    global EXITCODE
    migas_kwargs = {'status': 'C', 'status_desc': 'Success'}
    # `sys` will not have these attributes unless an error has been handled
    if hasattr(sys, 'last_type'):
        migas_kwargs = {
            'status': 'F',
            'status_desc': 'Finished with error(s)',
            'error_type': sys.last_type,
            'error_desc': sys.last_value,
        }
    elif EXITCODE != 0:
        migas_kwargs.update({'status': 'F', 'status_desc': f'Completed with exitcode {EXITCODE}'})

    send_breadcrumb(**migas_kwargs)


if __name__ == "__main__":
    main()
