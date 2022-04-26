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


def main():
    """Entry point for MRIQC's CLI."""
    import gc
    import os
    import sys
    from tempfile import mktemp
    from mriqc import config, messages
    from mriqc.cli.parser import parse_args

    # Run parser
    parse_args()

    _plugin = config.nipype.get_plugin()
    if config.nipype.plugin in ("MultiProc", "LegacyMultiProc"):
        from contextlib import suppress
        import multiprocessing as mp
        import multiprocessing.forkserver
        from mriqc.engine.plugin import MultiProcPlugin

        with suppress(RuntimeError):
            mp.set_start_method("forkserver")
            mp.forkserver.ensure_running()

        gc.collect()
        _plugin = {
            "plugin": MultiProcPlugin(plugin_args=config.nipype.plugin_args),
        }

    if config.execution.pdb:
        from mriqc.utils.debug import setup_exceptionhook

        setup_exceptionhook()

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
        from mriqc.workflows.core import init_mriqc_wf

        start_message = messages.PARTICIPANT_START.format(
            version=config.environment.version,
            bids_dir=config.execution.bids_dir,
            output_dir=config.execution.output_dir,
            analysis_level=config.workflow.analysis_level,
        )
        config.loggers.cli.log(25, start_message)
        mriqc_wf = init_mriqc_wf()
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
            mriqc_wf.run(**_plugin)

            # Warn about submitting measures AFTER
            if not config.execution.no_sub:
                config.loggers.cli.warning(config.DSA_MESSAGE)
        config.loggers.cli.log(25, messages.PARTICIPANT_FINISHED)

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


if __name__ == "__main__":
    main()
