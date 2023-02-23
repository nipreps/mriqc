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
"""Parser."""
import re

from mriqc import config


def _parse_participant_labels(value):
    """
    Drop ``sub-`` prefix of participant labels.

    >>> _parse_participant_labels("s060")
    ['s060']
    >>> _parse_participant_labels("sub-s060")
    ['s060']
    >>> _parse_participant_labels("s060 sub-s050")
    ['s050', 's060']
    >>> _parse_participant_labels("s060 sub-s060")
    ['s060']
    >>> _parse_participant_labels("s060\tsub-s060")
    ['s060']

    """
    return sorted(set(
        re.sub(r"^sub-", "", item.strip())
        for item in re.split(r"\s+", f"{value}".strip())
    ))


def _build_parser():
    """Build parser object."""
    import sys
    import warnings
    from argparse import Action, ArgumentDefaultsHelpFormatter, ArgumentParser
    from functools import partial
    from pathlib import Path
    from shutil import which

    from packaging.version import Version

    from .version import check_latest, is_flagged

    class DeprecateAction(Action):
        def __call__(self, parser, namespace, values, option_string=None):
            warnings.warn(f"Argument {option_string} is deprecated and is *ignored*.")
            delattr(namespace, self.dest)

    class ParticipantLabelAction(Action):
        def __call__(self, parser, namespace, values, option_string=None):
            setattr(namespace, self.dest, _parse_participant_labels(" ".join(values)))

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
        scale = {"G": 1, "T": 10**3, "M": 1e-3, "K": 1e-6, "B": 1e-9}
        digits = "".join([c for c in value if c.isdigit()])
        n_digits = len(digits)
        units = value[n_digits:] or "G"
        return int(digits) * scale[units[0]]

    def _bids_filter(value):
        from json import loads

        if value and Path(value).exists():
            return loads(Path(value).read_text())

    verstr = f"MRIQC v{config.environment.version}"
    currentv = Version(config.environment.version)

    parser = ArgumentParser(
        description=f"""\
MRIQC {config.environment.version}
Automated Quality Control and visual reports for Quality Assessment of structural \
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
        help="The root folder of a BIDS valid dataset (sub-XXXXX folders should "
        "be found at the top level in this folder).",
    )
    parser.add_argument(
        "output_dir",
        action="store",
        type=Path,
        help="The directory where the output files "
        "should be stored. If you are running group level analysis "
        "this folder should be prepopulated with the results of the "
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
        help="Increases log verbosity for each occurrence, debug level is -vvv.",
    )

    # TODO: add 'mouse', 'macaque', and other populations once the pipeline is working
    parser.add_argument(
        "--species",
        action="store",
        type=str,
        default="human",
        choices=["human", "rat"],
        help="Use appropriate template for population",
    )

    g_bids = parser.add_argument_group("Options for filtering BIDS queries")
    g_bids.add_argument(
        "--participant-label",
        "--participant_label",
        "--participant-labels",
        "--participant_labels",
        dest="participant_label",
        action=ParticipantLabelAction,
        nargs="+",
        help="A space delimited list of participant identifiers or a single "
        "identifier (the sub- prefix can be removed).",
    )
    g_bids.add_argument(
        "--session-id",
        action="store",
        nargs="*",
        type=str,
        help="Filter input dataset by session ID.",
    )
    g_bids.add_argument(
        "--run-id",
        action="store",
        type=int,
        nargs="*",
        help="Filter input dataset by run ID (only integer run IDs are valid).",
    )
    g_bids.add_argument(
        "--task-id",
        action="store",
        nargs="*",
        type=str,
        help="Filter input dataset by task ID.",
    )
    g_bids.add_argument(
        "-m",
        "--modalities",
        action="store",
        nargs="*",
        help="Filter input dataset by MRI type.",
    )
    g_bids.add_argument("--dsname", type=str, help="A dataset name.")
    g_bids.add_argument(
        "--bids-database-dir",
        metavar="PATH",
        help="Path to an existing PyBIDS database folder, for faster indexing "
        "(especially useful for large datasets).",
    )

    # General performance
    g_perfm = parser.add_argument_group("Options to handle performance")
    g_perfm.add_argument(
        "--nprocs",
        "--n_procs",
        "--n_cpus",
        "-n-cpus",
        action="store",
        type=PositiveInt,
        help="""\
Maximum number of simultaneously running parallel processes executed by *MRIQC* \
(e.g., several instances of ANTs' registration). \
However, when ``--nprocs`` is greater or equal to the ``--omp-nthreads`` option, \
it also sets the maximum number of threads that simultaneously running processes \
may aggregate (meaning, with ``--nprocs 16 --omp-nthreads 8`` a maximum of two \
8-CPU-threaded processes will be running at a given time). \
Under this mode of operation, ``--nprocs`` sets the maximum number of processors \
that can be assigned work within an *MRIQC* job, which includes all the processors \
used by currently running single- and multi-threaded processes. \
If ``None``, the number of CPUs available will be automatically assigned (which may \
not be what you want in, e.g., shared systems like a HPC cluster.""",
    )
    g_perfm.add_argument(
        "--omp-nthreads",
        "--ants-nthreads",
        action="store",
        type=PositiveInt,
        help="""\
Maximum number of threads that multi-threaded processes executed by *MRIQC* \
(e.g., ANTs' registration) can use. \
If ``None``, the number of CPUs available will be automatically assigned (which may \
not be what you want in, e.g., shared systems like a HPC cluster.""",
    )
    g_perfm.add_argument(
        "--mem",
        "--mem_gb",
        "--mem-gb",
        dest="memory_gb",
        action="store",
        type=_to_gb,
        help="Upper bound memory limit for MRIQC processes.",
    )
    g_perfm.add_argument(
        "--testing",
        dest="debug",
        action="store_true",
        default=False,
        help="Use testing settings for a minimal footprint.",
    )
    g_perfm.add_argument(
        "-f",
        "--float32",
        action="store_true",
        default=True,
        help="Cast the input data to float32 if it's represented in higher precision "
        "(saves space and improves performance).",
    )
    g_perfm.add_argument(
        "--pdb",
        dest="pdb",
        action="store_true",
        default=False,
        help="Open Python debugger (pdb) on exceptions.",
    )

    # Control instruments
    g_outputs = parser.add_argument_group("Instrumental options")
    g_outputs.add_argument(
        "-w",
        "--work-dir",
        action="store",
        type=Path,
        default=Path("work").absolute(),
        help="Path where intermediate results should be stored.",
    )
    g_outputs.add_argument("--verbose-reports", default=False, action="store_true")
    g_outputs.add_argument(
        "--write-graph",
        action="store_true",
        default=False,
        help="Write workflow graph.",
    )
    g_outputs.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Do not run the workflow.",
    )
    g_outputs.add_argument(
        "--resource-monitor",
        "--profile",
        dest="resource_monitor",
        action="store_true",
        default=False,
        help="Hook up the resource profiler callback to nipype.",
    )
    g_outputs.add_argument(
        "--use-plugin",
        action="store",
        default=None,
        type=Path,
        help="Nipype plugin configuration file.",
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
        help="IP address where the MRIQC WebAPI is listening.",
    )
    g_outputs.add_argument(
        "--webapi-port",
        action="store",
        type=int,
        help="Port where the MRIQC WebAPI is listening.",
    )

    g_outputs.add_argument(
        "--upload-strict",
        action="store_true",
        default=False,
        help="Upload will fail if upload is strict.",
    )
    g_outputs.add_argument(
        "--notrack",
        action="store_true",
        help="Opt-out of sending tracking information of this run to the NiPreps developers. This"
        " information helps to improve MRIQC and provides an indicator of real world usage "
        " crucial for obtaining funding.",
    )

    # ANTs options
    g_ants = parser.add_argument_group("Specific settings for ANTs")
    g_ants.add_argument(
        "--ants-float",
        action="store_true",
        default=False,
        help="Use float number precision on ANTs computations.",
    )
    g_ants.add_argument(
        "--ants-settings",
        action="store",
        help="Path to JSON file with settings for ANTs.",
    )

    # Functional workflow settings
    g_func = parser.add_argument_group("Functional MRI workflow configuration")
    if which("melodic") is not None:
        g_func.add_argument(
            "--ica",
            action="store_true",
            default=False,
            help="Run ICA on the raw data and include the components "
            "in the individual reports (slow but potentially very insightful).",
        )
    g_func.add_argument(
        "--fft-spikes-detector",
        action="store_true",
        default=False,
        help="Turn on FFT based spike detector (slow).",
    )
    g_func.add_argument(
        "--fd_thres",
        action="store",
        default=0.2,
        type=float,
        help="Threshold on framewise displacement estimates to detect outliers.",
    )
    g_func.add_argument(
        "--deoblique",
        action="store_true",
        default=False,
        help="Deoblique the functional scans during head motion correction "
        "preprocessing.",
    )
    g_func.add_argument(
        "--despike",
        action="store_true",
        default=False,
        help="Despike the functional scans during head motion correction "
        "preprocessing.",
    )
    g_func.add_argument(
        "--start-idx",
        action=DeprecateAction,
        type=int,
        help="DEPRECATED Initial volume in functional timeseries that should be "
        "considered for preprocessing.",
    )
    g_func.add_argument(
        "--stop-idx",
        action=DeprecateAction,
        type=int,
        help="DEPRECATED Final volume in functional timeseries that should be "
        "considered for preprocessing.",
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
    from contextlib import suppress

    from ..utils.bids import collect_bids_data

    parser = _build_parser()
    opts = parser.parse_args(args, namespace)
    config.execution.log_level = int(max(25 - 5 * opts.verbose_count, DEBUG))
    config.from_dict(vars(opts))

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

    # Setup directories
    config.execution.log_dir = output_dir / "logs"
    # Check and create output and working directories
    config.execution.log_dir.mkdir(exist_ok=True, parents=True)
    output_dir.mkdir(exist_ok=True, parents=True)
    work_dir.mkdir(exist_ok=True, parents=True)

    # Force initialization of the BIDSLayout
    config.execution.init()

    participant_label = config.execution.layout.get_subjects()
    if config.execution.participant_label is not None:
        selected_label = set(config.execution.participant_label)
        missing_subjects = selected_label - set(participant_label)
        if missing_subjects:
            parser.error(
                "One or more participant labels were not found in the BIDS directory: "
                f"{', '.join(missing_subjects)}."
            )
        participant_label = selected_label

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

    # Estimate the biggest file size / leave 1GB if some file does not exist (datalad)
    with suppress(FileNotFoundError):
        config.workflow.biggest_file_gb = _get_biggest_file_size_gb(
            [i for sublist in config.workflow.inputs.values() for i in sublist]
        )

    # set specifics for alternative populations
    if opts.species.lower() != "human":
        config.workflow.species = opts.species
        # TODO: add other species once rats are working
        if opts.species.lower() == "rat":
            config.workflow.template_id = "Fischer344"
            # mean distance from the lateral edge to the center of the brain is
            # ~ PA:10 mm, LR:7.5 mm, and IS:5 mm (see DOI: 10.1089/089771503770802853)
            # roll movement is most likely to occur, so set to 7.5 mm
            config.workflow.fd_radius = 7.5
            config.workflow.headmask = "NoBET"
            # block uploads for the moment; can be reversed before wider release
            config.execution.no_sub = True


def _get_biggest_file_size_gb(files):
    import os

    max_size = 0
    for file in files:
        size = os.path.getsize(file) / (1024**3)
        if size > max_size:
            max_size = size
    return max_size
