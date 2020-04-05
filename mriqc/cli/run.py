#!/usr/bin/env python
"""MRIQC run script."""
from .. import config


def main():
    """Entry point."""
    import os
    import sys
    import gc
    from multiprocessing import Process, Manager
    from .parser import parse_args

    # Run parser
    parse_args()

    # CRITICAL Save the config to a file. This is necessary because the execution graph
    # is built as a separate process to keep the memory footprint low. The most
    # straightforward way to communicate with the child process is via the filesystem.
    config_file = config.execution.work_dir / ".mriqc.toml"
    config.to_filename(config_file)

    # Set up participant level
    if "participant" in config.workflow.analysis_level:
        config.loggers.cli.log(
            25,
            f"""
    Running MRIQC version {config.environment.version}:
      * BIDS dataset path: {config.execution.bids_dir}.
      * Output folder: {config.execution.output_dir}.
      * Analysis levels: {config.workflow.analysis_level}.
""",
        )
        # CRITICAL Call build_workflow(config_file, retval) in a subprocess.
        # Because Python on Linux does not ever free virtual memory (VM), running the
        # workflow construction jailed within a process preempts excessive VM buildup.
        with Manager() as mgr:
            from .workflow import build_workflow

            retval = mgr.dict()
            p = Process(target=build_workflow, args=(str(config_file), retval))
            p.start()
            p.join()

            mriqc_wf = retval.get("workflow", None)
            retcode = p.exitcode or retval.get("return_code", 0)

        # CRITICAL Load the config from the file. This is necessary because the ``build_workflow``
        # function executed constrained in a process may change the config (and thus the global
        # state of MRIQC).
        config.load(config_file)

        retcode = retcode or (mriqc_wf is None) * os.EX_SOFTWARE
        if retcode != 0:
            sys.exit(retcode)

        if mriqc_wf and config.execution.write_graph:
            mriqc_wf.write_graph(graph2use="colored", format="svg", simple_form=True)

        # Clean up master process before running workflow, which may create forks
        gc.collect()

        if not config.execution.dry_run:
            # Warn about submitting measures BEFORE
            if not config.execution.no_sub:
                config.loggers.cli.warning(config.DSA_MESSAGE)

            # run MRIQC
            mriqc_wf.run(**config.nipype.get_plugin())

            # Warn about submitting measures AFTER
            if not config.execution.no_sub:
                config.loggers.cli.warning(config.DSA_MESSAGE)
        config.loggers.cli.log(25, "Participant level finished successfully.")

    # Set up group level
    if "group" in config.workflow.analysis_level:
        from ..utils.bids import DEFAULT_TYPES
        from ..reports import group_html
        from ..utils.misc import generate_tsv  # , generate_pred

        config.loggers.cli.info("Group level started...")

        # Generate reports
        mod_group_reports = []
        for mod in config.execution.modalities or DEFAULT_TYPES:
            output_dir = config.execution.output_dir
            dataframe, out_tsv = generate_tsv(output_dir, mod)
            # If there are no iqm.json files, nothing to do.
            if dataframe is None:
                continue

            config.loggers.cli.info(
                f"Generated summary TSV table for the {mod} data ({out_tsv})"
            )

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
            config.loggers.cli.info(f"Group-{mod} report generated ({out_html})")
            mod_group_reports.append(mod)

        if not mod_group_reports:
            raise Exception("No data found. No group level reports were generated.")

        config.loggers.cli.info("Group level finished successfully.")


if __name__ == "__main__":
    main()
