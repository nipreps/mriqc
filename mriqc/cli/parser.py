# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Parser."""
from .. import config


def _build_parser():
    """Build parser object."""
    import sys
    from functools import partial
    from pathlib import Path
    from argparse import (
        ArgumentParser,
        ArgumentDefaultsHelpFormatter,
    )
    from packaging.version import Version
    from .version import check_latest, is_flagged

    def _path_exists(path, parser):
        """Ensure a given path exists."""
        if path is None or not Path(path).exists():
            raise parser.error(f"Path does not exist: <{path}>.")
        return Path(path).expanduser().absolute()

    def _min_one(value, parser):
        """Ensure an argument is not lower than 1."""
        value = int(value)
        if value < 1:
            raise parser.error("Argument can't be less than one.")
        return value

    def _to_gb(value):
        scale = {"G": 1, "T": 10 ** 3, "M": 1e-3, "K": 1e-6, "B": 1e-9}
        digits = "".join([c for c in value if c.isdigit()])
        units = value[len(digits):] or "G"
        return int(digits) * scale[units[0]]

    def _drop_sub(value):
        value = str(value)
        return value.lstrip("sub-")

    def _bids_filter(value):
        from json import loads

        if value and Path(value).exists():
            return loads(Path(value).read_text())

    verstr = f"MRIQC v{config.environment.version}"
    currentv = Version(config.environment.version)

    parser = ArgumentParser(
        description=f"""\
MRIQC {config.environment.version}
Automated Quality Control and visual reports for Quality Assesment of structural \
(T1w, T2w) and functional MRI of the brain.

{config.DSA_MESSAGE}""",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    PathExists = partial(_path_exists, parser=parser)
    PositiveInt = partial(_min_one, parser=parser)

    # Arguments as specified by BIDS-Apps
    # required, positional arguments
    # IMPORTANT: they must go directly with the parser object
    parser.add_argument(
        "bids_dir",
        action="store",
        type=PathExists,
        help="the root folder of a BIDS valid dataset (sub-XXXXX folders should "
        "be found at the top level in this folder).",
    )
    parser.add_argument(
        "output_dir",
        action="store",
        type=Path,
        help="The directory where the output files "
        "should be stored. If you are running group level analysis "
        "this folder should be prepopulated with the results of the"
        "participant level analysis.",
    )
    parser.add_argument(
        "analysis_level",
        action="store",
        nargs="+",
        help="Level of the analysis that will be performed. "
        "Multiple participant level analyses can be run independently "
        "(in parallel) using the same output_dir.",
        choices=["participant", "group"],
    )

    # optional arguments
    parser.add_argument("--version", action="version", version=verstr)
    parser.add_argument(
        "-v",
        "--verbose",
        dest="verbose_count",
        action="count",
        default=0,
        help="increases log verbosity for each occurrence, debug level is -vvv",
    )

    g_bids = parser.add_argument_group("Options for filtering BIDS queries")
    g_bids.add_argument(
        "--participant-label",
        "--participant_label",
        action="store",
        nargs="+",
        type=_drop_sub,
        help="a space delimited list of participant identifiers or a single "
        "identifier (the sub- prefix can be removed)",
    )
    g_bids.add_argument(
        "--session-id",
        action="store",
        nargs="*",
        type=str,
        help="filter input dataset by session id",
    )
    g_bids.add_argument(
        "--run-id",
        action="store",
        type=int,
        nargs="*",
        help="filter input dataset by run id (only integer run ids are valid)",
    )
    g_bids.add_argument(
        "--task-id",
        action="store",
        nargs="*",
        type=str,
        help="filter input dataset by task id",
    )
    g_bids.add_argument(
        "-m",
        "--modalities",
        action="store",
        nargs="*",
        help="filter input dataset by MRI type",
    )
    g_bids.add_argument("--dsname", type=str, help="a dataset name")

    # General performance
    g_perfm = parser.add_argument_group("Options to handle performance")
    g_perfm.add_argument(
        "--nprocs",
        "--n_procs",
        "--n_cpus",
        "-n-cpus",
        action="store",
        type=PositiveInt,
        help="maximum number of threads across all processes",
    )
    g_perfm.add_argument(
        "--omp-nthreads",
        "--ants-nthreads",
        action="store",
        type=PositiveInt,
        help="maximum number of threads per-process",
    )
    g_perfm.add_argument(
        "--mem",
        "--mem_gb",
        "--mem-gb",
        dest="memory_gb",
        action="store",
        type=_to_gb,
        help="upper bound memory limit for MRIQC processes",
    )
    g_perfm.add_argument(
        "--testing",
        dest="debug",
        action="store_true",
        default=False,
        help="use testing settings for a minimal footprint",
    )
    g_perfm.add_argument(
        "-f",
        "--float32",
        action="store_true",
        default=True,
        help="Cast the input data to float32 if it's represented in higher precision "
        "(saves space and improves perfomance)",
    )

    # Control instruments
    g_outputs = parser.add_argument_group("Instrumental options")
    g_outputs.add_argument(
        "-w",
        "--work-dir",
        action="store",
        type=Path,
        default=Path("work").absolute(),
        help="path where intermediate results should be stored",
    )
    g_outputs.add_argument("--verbose-reports", default=False, action="store_true")
    g_outputs.add_argument(
        "--write-graph",
        action="store_true",
        default=False,
        help="Write workflow graph.",
    )
    g_outputs.add_argument(
        "--dry-run", action="store_true", default=False, help="Do not run the workflow."
    )
    g_outputs.add_argument(
        "--profile",
        action="store_true",
        default=False,
        help="hook up the resource profiler callback to nipype",
    )
    g_outputs.add_argument(
        "--use-plugin",
        action="store",
        default=None,
        type=Path,
        help="nipype plugin configuration file",
    )
    g_outputs.add_argument(
        "--no-sub",
        default=False,
        action="store_true",
        help="Turn off submission of anonymized quality metrics "
        "to MRIQC's metrics repository.",
    )
    g_outputs.add_argument(
        "--email",
        action="store",
        default="",
        type=str,
        help="Email address to include with quality metric submission.",
    )

    g_outputs.add_argument(
        "--webapi-url",
        action="store",
        type=str,
        help="IP address where the MRIQC WebAPI is listening",
    )
    g_outputs.add_argument(
        "--webapi-port",
        action="store",
        type=int,
        help="port where the MRIQC WebAPI is listening",
    )

    g_outputs.add_argument(
        "--upload-strict",
        action="store_true",
        default=False,
        help="upload will fail if if upload is strict",
    )

    # Workflow settings
    g_conf = parser.add_argument_group("Workflow configuration")
    g_conf.add_argument(
        "--ica",
        action="store_true",
        default=False,
        help="Run ICA on the raw data and include the components "
        "in the individual reports (slow but potentially very insightful)",
    )
    g_conf.add_argument(
        "--hmc-afni",
        action="store_true",
        default=True,
        help="Use AFNI 3dvolreg for head motion correction (HMC) - default",
    )
    g_conf.add_argument(
        "--hmc-fsl",
        action="store_true",
        default=False,
        help="Use FSL MCFLIRT instead of AFNI for head motion correction (HMC)",
    )
    g_conf.add_argument(
        "--fft-spikes-detector",
        action="store_true",
        default=False,
        help="Turn on FFT based spike detector (slow).",
    )
    g_conf.add_argument(
        "--fd_thres",
        action="store",
        default=0.2,
        type=float,
        help="Threshold on Framewise Displacement estimates to detect outliers.",
    )

    # ANTs options
    g_ants = parser.add_argument_group("Specific settings for ANTs")
    g_ants.add_argument(
        "--ants-float",
        action="store_true",
        default=False,
        help="use float number precision on ANTs computations",
    )
    g_ants.add_argument(
        "--ants-settings",
        action="store",
        help="path to JSON file with settings for ANTS",
    )

    # AFNI head motion correction settings
    g_afni = parser.add_argument_group("Specific settings for AFNI")
    g_afni.add_argument(
        "--deoblique",
        action="store_true",
        default=False,
        help="Deoblique the functional scans during head motion "
        "correction preprocessing",
    )
    g_afni.add_argument(
        "--despike",
        action="store_true",
        default=False,
        help="Despike the functional scans during head motion correction "
        "preprocessing",
    )
    g_afni.add_argument(
        "--start-idx",
        action="store",
        type=int,
        help="Initial volume in functional timeseries that should be "
        "considered for preprocessing",
    )
    g_afni.add_argument(
        "--stop-idx",
        action="store",
        type=int,
        help="Final volume in functional timeseries that should be "
        "considered for preprocessing",
    )
    g_afni.add_argument(
        "--correct-slice-timing",
        action="store_true",
        default=False,
        help="Perform slice timing correction",
    )

    latest = check_latest()
    if latest is not None and currentv < latest:
        print(
            f"""\
You are using MRIQC v{currentv}, and a newer version is available: {latest}.""",
            file=sys.stderr,
        )

    _blist = is_flagged()
    if _blist[0]:
        _reason = _blist[1] or "unknown"
        print(
            f"""\
WARNING: This version of MRIQC ({config.environment.version}) has been FLAGGED
(reason: {_reason}).
That means some severe flaw was found in it and we strongly \
discourage its usage.""",
            file=sys.stderr,
        )

    return parser


def parse_args(args=None, namespace=None):
    """Parse args and run further checks on the command line."""
    from logging import DEBUG
    from ..utils.bids import collect_bids_data

    parser = _build_parser()
    opts = parser.parse_args(args, namespace)
    config.execution.log_level = int(max(25 - 5 * opts.verbose_count, DEBUG))
    config.from_dict(vars(opts))
    config.loggers.init()

    # Load base plugin_settings from file if --use-plugin
    if opts.use_plugin is not None:
        from yaml import load as loadyml

        with open(opts.use_plugin) as f:
            plugin_settings = loadyml(f)
        _plugin = plugin_settings.get("plugin")
        if _plugin:
            config.nipype.plugin = _plugin
            config.nipype.plugin_args = plugin_settings.get("plugin_args", {})
            config.nipype.nprocs = config.nipype.plugin_args.get(
                "nprocs", config.nipype.nprocs
            )

    # Resource management options
    # Note that we're making strong assumptions about valid plugin args
    # This may need to be revisited if people try to use batch plugins
    if 1 < config.nipype.nprocs < config.nipype.omp_nthreads:
        config.loggers.cli.warning(
            "Per-process threads (--omp-nthreads=%d) exceed total "
            "threads (--nthreads/--n_cpus=%d)",
            config.nipype.omp_nthread,
            config.nipype.nprocs,
        )

    bids_dir = config.execution.bids_dir
    output_dir = config.execution.output_dir
    work_dir = config.execution.work_dir
    version = config.environment.version

    # Ensure input and output folders are not the same
    if output_dir == bids_dir:
        parser.error(
            "The selected output folder is the same as the input BIDS folder. "
            "Please modify the output path (suggestion: %s)."
            % bids_dir
            / "derivatives"
            / ("mriqc-%s" % version.split("+")[0])
        )

    if bids_dir in work_dir.parents:
        parser.error(
            "The selected working directory is a subdirectory of the input BIDS folder. "
            "Please modify the output path."
        )

    # Validate inputs
    # if not opts.skip_bids_validation:
    #     from ..utils.bids import validate_input_dir

    #     build_log.info(
    #         "Making sure the input data is BIDS compliant (warnings can be ignored in most "
    #         "cases)."
    #     )
    #     validate_input_dir(
    #         config.environment.exec_env, opts.bids_dir, opts.participant_label
    #     )

    # Setup directories
    config.execution.log_dir = output_dir / "logs"
    # Check and create output and working directories
    config.execution.log_dir.mkdir(exist_ok=True, parents=True)
    output_dir.mkdir(exist_ok=True, parents=True)
    work_dir.mkdir(exist_ok=True, parents=True)

    # Force initialization of the BIDSLayout
    config.execution.init()
    all_subjects = config.execution.layout.get_subjects()
    if config.execution.participant_label is None:
        config.execution.participant_label = all_subjects

    participant_label = set(config.execution.participant_label)
    missing_subjects = participant_label - set(all_subjects)
    if missing_subjects:
        parser.error(
            "One or more participant labels were not found in the BIDS directory: "
            f"{', '.join(missing_subjects)}."
        )

    config.execution.participant_label = sorted(participant_label)

    # Handle analysis_level
    analysis_level = set(config.workflow.analysis_level)
    if not config.execution.participant_label:
        analysis_level.add("group")
    config.workflow.analysis_level = list(analysis_level)

    # List of files to be run
    bids_filters = {
        "participant_label": config.execution.participant_label,
        "session": config.execution.session_id,
        "run": config.execution.run_id,
        "task": config.execution.task_id,
        "bids_type": config.execution.modalities,
    }
    config.workflow.inputs = {
        mod: files
        for mod, files in collect_bids_data(
            config.execution.layout, **bids_filters
        ).items()
        if files
    }

    # Check the query is not empty
    if not list(config.workflow.inputs.values()):
        _j = "\n *"
        parser.error(
            f"""\
Querying BIDS dataset at <{config.execution.bids_dir}> got an empty result.
Please, check out your currently set filters:
{_j.join([''] + [': '.join((k, str(v))) for k, v in bids_filters.items()])}"""
        )

    # Check no DWI or others are sneaked into MRIQC
    unknown_mods = set(config.workflow.inputs.keys()) - set(("T1w", "T2w", "bold"))
    if unknown_mods:
        parser.error(
            "MRIQC is unable to process the following modalities: "
            f'{", ".join(unknown_mods)}.'
        )

    # Estimate the biggest file size
    config.workflow.biggest_file_gb = _get_biggest_file_size_gb(
        [i for sublist in config.workflow.inputs.values() for i in sublist]
    )


def _get_biggest_file_size_gb(files):
    import os

    max_size = 0
    for file in files:
        size = os.path.getsize(file) / (1024 ** 3)
        if size > max_size:
            max_size = size
    return max_size
