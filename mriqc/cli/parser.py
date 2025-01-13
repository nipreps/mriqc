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
    return sorted(
        {re.sub(r'^sub-', '', item.strip()) for item in re.split(r'\s+', f'{value}'.strip())}
    )


def _build_parser():
    """Build parser object."""
    import sys
    import warnings
    from argparse import Action, ArgumentDefaultsHelpFormatter, ArgumentParser
    from functools import partial
    from pathlib import Path

    from packaging.version import Version

    from .version import check_latest, is_flagged

    class DeprecateAction(Action):
        def __call__(self, parser, namespace, values, option_string=None):
            warnings.warn(
                f'Argument {option_string} is deprecated and is *ignored*.',
                stacklevel=2,
            )
            delattr(namespace, self.dest)

    class ParticipantLabelAction(Action):
        def __call__(self, parser, namespace, values, option_string=None):
            setattr(namespace, self.dest, _parse_participant_labels(' '.join(values)))

    def _path_exists(path, parser):
        """Ensure a given path exists."""
        if path is None or not Path(path).exists():
            raise parser.error(f'Path does not exist: <{path}>.')
        return Path(path).expanduser().absolute()

    def _min_one(value, parser):
        """Ensure an argument is not lower than 1."""
        value = int(value)
        if value < 1:
            raise parser.error("Argument can't be less than one.")
        return value

    def _to_gb(value):
        scale = {'G': 1, 'T': 10**3, 'M': 1e-3, 'K': 1e-6, 'B': 1e-9}
        digits = ''.join([c for c in value if c.isdigit()])
        n_digits = len(digits)
        units = value[n_digits:] or 'G'
        return int(digits) * scale[units[0]]

    def _bids_filter(value):
        from json import loads

        if value and Path(value).exists():
            return loads(Path(value).read_text())

    verstr = f'MRIQC v{config.environment.version}'
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
        'bids_dir',
        action='store',
        type=PathExists,
        help='The root folder of a BIDS valid dataset (sub-XXXXX folders should '
        'be found at the top level in this folder).',
    )
    parser.add_argument(
        'output_dir',
        action='store',
        type=Path,
        help='The directory where the output files '
        'should be stored. If you are running group level analysis '
        'this folder should be prepopulated with the results of the '
        'participant level analysis.',
    )
    parser.add_argument(
        'analysis_level',
        action='store',
        nargs='+',
        help='Level of the analysis that will be performed. '
        'Multiple participant level analyses can be run independently '
        '(in parallel) using the same output_dir.',
        choices=['participant', 'group'],
    )

    # optional arguments
    parser.add_argument('--version', action='version', version=verstr)
    parser.add_argument(
        '-v',
        '--verbose',
        dest='verbose_count',
        action='count',
        default=0,
        help='Increases log verbosity for each occurrence, debug level is -vvv.',
    )

    # TODO: add 'mouse', 'macaque', and other populations once the pipeline is working
    parser.add_argument(
        '--species',
        action='store',
        type=str,
        default='human',
        choices=['human', 'rat'],
        help='Use appropriate template for population',
    )

    g_bids = parser.add_argument_group('Options for filtering BIDS queries')
    g_bids.add_argument(
        '--participant-label',
        '--participant_label',
        '--participant-labels',
        '--participant_labels',
        dest='participant_label',
        action=ParticipantLabelAction,
        nargs='+',
        help='A space delimited list of participant identifiers or a single '
        'identifier (the sub- prefix can be removed).',
    )
    g_bids.add_argument(
        '--bids-filter-file',
        action='store',
        type=Path,
        metavar='PATH',
        help='a JSON file describing custom BIDS input filter using pybids '
        '{<suffix>:{<entity>:<filter>,...},...} '
        '(https://github.com/bids-standard/pybids/blob/master/bids/layout/config/bids.json)',
    )
    g_bids.add_argument(
        '--session-id',
        action='store',
        nargs='*',
        type=str,
        help='Filter input dataset by session ID.',
    )
    g_bids.add_argument(
        '--run-id',
        action='store',
        type=int,
        nargs='*',
        help='DEPRECATED - This argument will be disabled. Use ``--bids-filter-file`` instead.',
    )
    g_bids.add_argument(
        '--task-id',
        action='store',
        nargs='*',
        type=str,
        help='Filter input dataset by task ID.',
    )
    g_bids.add_argument(
        '-m',
        '--modalities',
        action='store',
        choices=config.SUPPORTED_SUFFIXES,
        default=config.SUPPORTED_SUFFIXES,
        nargs='*',
        help='Filter input dataset by MRI type.',
    )
    g_bids.add_argument('--dsname', type=str, help='A dataset name.')
    g_bids.add_argument(
        '--bids-database-dir',
        metavar='PATH',
        help='Path to an existing PyBIDS database folder, for faster indexing '
        '(especially useful for large datasets).',
    )
    g_bids.add_argument(
        '--bids-database-wipe',
        action='store_true',
        default=False,
        help='Wipe out previously existing BIDS indexing caches, forcing re-indexing.',
    )
    g_bids.add_argument(
        '--no-datalad-get',
        action='store_false',
        dest='datalad_get',
        help='Disable attempting to get remote files in DataLad datasets.',
    )

    # General performance
    g_perfm = parser.add_argument_group('Options to handle performance')
    g_perfm.add_argument(
        '--nprocs',
        '--n_procs',
        '--n_cpus',
        '-n-cpus',
        action='store',
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
        '--omp-nthreads',
        '--ants-nthreads',
        action='store',
        type=PositiveInt,
        help="""\
Maximum number of threads that multi-threaded processes executed by *MRIQC* \
(e.g., ANTs' registration) can use. \
If ``None``, the number of CPUs available will be automatically assigned (which may \
not be what you want in, e.g., shared systems like a HPC cluster.""",
    )
    g_perfm.add_argument(
        '--mem',
        '--mem_gb',
        '--mem-gb',
        dest='memory_gb',
        action='store',
        type=_to_gb,
        help='Upper bound memory limit for MRIQC processes.',
    )
    g_perfm.add_argument(
        '--testing',
        dest='debug',
        action='store_true',
        default=False,
        help='Use testing settings for a minimal footprint.',
    )
    g_perfm.add_argument(
        '-f',
        '--float32',
        action='store_true',
        default=True,
        help="Cast the input data to float32 if it's represented in higher precision "
        '(saves space and improves performance).',
    )
    g_perfm.add_argument(
        '--pdb',
        dest='pdb',
        action='store_true',
        default=False,
        help='Open Python debugger (pdb) on exceptions.',
    )

    # Control instruments
    g_outputs = parser.add_argument_group('Instrumental options')
    g_outputs.add_argument(
        '-w',
        '--work-dir',
        action='store',
        type=Path,
        default=Path('work').absolute(),
        help='Path where intermediate results should be stored.',
    )
    g_outputs.add_argument('--verbose-reports', default=False, action='store_true')
    g_outputs.add_argument('--reports-only', default=False, action='store_true')
    g_outputs.add_argument(
        '--write-graph',
        action='store_true',
        default=False,
        help='Write workflow graph.',
    )
    g_outputs.add_argument(
        '--dry-run',
        action='store_true',
        default=False,
        help='Do not run the workflow.',
    )
    g_outputs.add_argument(
        '--resource-monitor',
        '--profile',
        dest='resource_monitor',
        action='store_true',
        default=False,
        help='Hook up the resource profiler callback to nipype.',
    )
    g_outputs.add_argument(
        '--use-plugin',
        action='store',
        default=None,
        type=Path,
        help='Nipype plugin configuration file.',
    )
    g_outputs.add_argument(
        '--crashfile-format',
        action='store',
        default='txt',
        choices=['txt', 'pklz'],
        type=str,
        help='Nipype crashfile format',
    )
    g_outputs.add_argument(
        '--no-sub',
        default=False,
        action='store_true',
        help="Turn off submission of anonymized quality metrics to MRIQC's metrics repository.",
    )
    g_outputs.add_argument(
        '--email',
        action='store',
        default='',
        type=str,
        help='Email address to include with quality metric submission.',
    )

    g_outputs.add_argument(
        '--webapi-url',
        action='store',
        type=str,
        help='IP address where the MRIQC WebAPI is listening.',
    )
    g_outputs.add_argument(
        '--webapi-port',
        action='store',
        type=int,
        help='Port where the MRIQC WebAPI is listening.',
    )

    g_outputs.add_argument(
        '--upload-strict',
        action='store_true',
        default=False,
        help='Upload will fail if upload is strict.',
    )
    g_outputs.add_argument(
        '--notrack',
        action='store_true',
        help='Opt-out of sending tracking information of this run to the NiPreps developers. This'
        ' information helps to improve MRIQC and provides an indicator of real world usage '
        ' crucial for obtaining funding.',
    )

    # ANTs options
    g_ants = parser.add_argument_group('Specific settings for ANTs')
    g_ants.add_argument(
        '--ants-float',
        action='store_true',
        default=False,
        help='Use float number precision on ANTs computations.',
    )
    g_ants.add_argument(
        '--ants-settings',
        action='store',
        help='Path to JSON file with settings for ANTs.',
    )

    # Diffusion workflow settings
    g_dwi = parser.add_argument_group('Diffusion MRI workflow configuration')
    g_dwi.add_argument(
        '--min-dwi-length',
        action='store',
        default=config.workflow.min_len_dwi,
        dest='min_len_dwi',
        help='Drop DWI runs with fewer orientations than this threshold.',
        type=int,
    )

    # Functional workflow settings
    g_func = parser.add_argument_group('Functional MRI workflow configuration')
    g_func.add_argument(
        '--min-bold-length',
        action='store',
        default=config.workflow.min_len_bold,
        dest='min_len_bold',
        help='Drop BOLD runs with fewer time points than this threshold.',
        type=int,
    )
    g_func.add_argument(
        '--fft-spikes-detector',
        action='store_true',
        default=False,
        help='Turn on FFT based spike detector (slow).',
    )
    g_func.add_argument(
        '--fd_thres',
        action='store',
        default=0.2,
        type=float,
        help='Threshold on framewise displacement estimates to detect outliers.',
    )
    g_func.add_argument(
        '--deoblique',
        action='store_true',
        default=False,
        help='Deoblique the functional scans during head motion correction preprocessing.',
    )
    g_func.add_argument(
        '--despike',
        action='store_true',
        default=False,
        help='Despike the functional scans during head motion correction preprocessing.',
    )
    g_func.add_argument(
        '--start-idx',
        action=DeprecateAction,
        type=int,
        help='DEPRECATED Initial volume in functional timeseries that should be '
        'considered for preprocessing.',
    )
    g_func.add_argument(
        '--stop-idx',
        action=DeprecateAction,
        type=int,
        help='DEPRECATED Final volume in functional timeseries that should be '
        'considered for preprocessing.',
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
        _reason = _blist[1] or 'unknown'
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
    from json import loads
    from logging import DEBUG, FileHandler
    from pathlib import Path
    from pprint import pformat

    from niworkflows.utils.bids import DEFAULT_BIDS_QUERIES, collect_data

    from mriqc import __version__, data
    from mriqc._warnings import DATE_FMT, LOGGER_FMT, _LogFormatter
    from mriqc.messages import PARTICIPANT_START
    from mriqc.utils.misc import initialize_meta_and_data

    parser = _build_parser()
    opts = parser.parse_args(args, namespace)
    config.execution.log_level = int(max(25 - 5 * opts.verbose_count, DEBUG))

    config.loggers.init()

    _log_file = Path(opts.output_dir) / 'logs' / f'mriqc-{config.execution.run_uuid}.log'
    _log_file.parent.mkdir(exist_ok=True, parents=True)
    _handler = FileHandler(_log_file)
    _handler.setFormatter(
        _LogFormatter(
            fmt=LOGGER_FMT.format(color='', reset=''),
            datefmt=DATE_FMT,
            colored=False,
        )
    )
    config.loggers.default.addHandler(_handler)

    extra_messages = ['']

    if opts.bids_filter_file:
        extra_messages.insert(
            0,
            f'  * BIDS filters-file: {opts.bids_filter_file.absolute()}.',
        )

    notice_path = data.load.readable('NOTICE')
    config.loggers.cli.log(
        26,
        PARTICIPANT_START.format(
            version=__version__,
            bids_dir=opts.bids_dir,
            output_dir=opts.output_dir,
            analysis_level=opts.analysis_level,
            notice='\n  '.join(
                ['NOTICE'] + notice_path.read_text().splitlines(keepends=False)[1:]
            ),
            extra_messages='\n'.join(extra_messages),
        ),
    )
    config.from_dict(vars(opts))

    # Load base plugin_settings from file if --use-plugin
    if opts.use_plugin is not None:
        from yaml import safe_load as loadyml

        with open(opts.use_plugin) as f:
            plugin_settings = loadyml(f)
        _plugin = plugin_settings.get('plugin')
        if _plugin:
            config.nipype.plugin = _plugin
            config.nipype.plugin_args = plugin_settings.get('plugin_args', {})
            config.nipype.nprocs = config.nipype.plugin_args.get('nprocs', config.nipype.nprocs)

    # Load BIDS filters
    if opts.bids_filter_file:
        config.execution.bids_filters = {
            k.lower(): v for k, v in loads(opts.bids_filter_file.read_text()).items()
        }

    bids_dir = config.execution.bids_dir
    output_dir = config.execution.output_dir
    work_dir = config.execution.work_dir
    version = config.environment.version

    # Ensure input and output folders are not the same
    if output_dir == bids_dir:
        parser.error(
            'The selected output folder is the same as the input BIDS folder. '
            f'Please modify the output path (suggestion: {bids_dir}).'
            / 'derivatives'
            / ('mriqc-{}'.format(version.split('+')[0]))
        )

    if bids_dir in work_dir.parents:
        parser.error(
            'The selected working directory is a subdirectory of the input BIDS folder. '
            'Please modify the output path.'
        )

    config.execution.bids_dir_datalad = (
        config.execution.datalad_get
        and (bids_dir / '.git').exists()
        and (bids_dir / '.datalad').exists()
    )

    # Setup directories
    config.execution.log_dir = output_dir / 'logs'
    # Check and create output and working directories
    config.execution.log_dir.mkdir(exist_ok=True, parents=True)
    output_dir.mkdir(exist_ok=True, parents=True)
    work_dir.mkdir(exist_ok=True, parents=True)

    # Force initialization of the BIDSLayout
    config.execution.init()

    participant_label = [
        d.name[4:] for d in config.execution.bids_dir.glob('sub-*') if d.is_dir() and d.exists()
    ]

    if config.execution.participant_label is not None:
        selected_label = set(config.execution.participant_label)
        if missing_subjects := selected_label - set(participant_label):
            parser.error(
                'One or more participant labels were not found in the BIDS directory: '
                f'{", ".join(missing_subjects)}.'
            )
        participant_label = selected_label

    config.execution.participant_label = sorted(participant_label)

    # Handle analysis_level
    analysis_level = set(config.workflow.analysis_level)
    if not config.execution.participant_label:
        analysis_level.add('group')
    config.workflow.analysis_level = list(analysis_level)

    # List of files to be run
    lc_modalities = [mod.lower() for mod in config.execution.modalities]
    bids_dataset, _ = collect_data(
        config.execution.layout,
        config.execution.participant_label,
        session_id=config.execution.session_id,
        task=config.execution.task_id,
        group_echos=True,
        bids_filters={mod: config.execution.bids_filters.get(mod, {}) for mod in lc_modalities},
        queries={mod: DEFAULT_BIDS_QUERIES[mod] for mod in lc_modalities},
    )

    # Drop empty queries
    bids_dataset = {mod: files for mod, files in bids_dataset.items() if files}
    config.workflow.inputs = bids_dataset

    # Check the query is not empty
    if not list(config.workflow.inputs.values()):
        ffile = (
            '(--bids-filter-file was not set)'
            if not opts.bids_filter_file
            else f"(with '--bids-filter-file {opts.bids_filter_file}')"
        )
        parser.error(
            f"""\
Querying BIDS dataset at <{config.execution.bids_dir}> got an empty result.
Please, check out your currently set filters {ffile}:
{pformat(config.execution.bids_filters, indent=2, width=99)}"""
        )

    # Check no DWI or others are sneaked into MRIQC
    unknown_mods = set(config.workflow.inputs.keys()) - {
        suffix.lower() for suffix in config.SUPPORTED_SUFFIXES
    }
    if unknown_mods:
        parser.error(
            f'MRIQC is unable to process the following modalities: {", ".join(unknown_mods)}.'
        )

    initialize_meta_and_data()

    # set specifics for alternative populations
    if opts.species.lower() != 'human':
        config.workflow.species = opts.species
        # TODO: add other species once rats are working
        if opts.species.lower() == 'rat':
            config.workflow.template_id = 'Fischer344'
            # mean distance from the lateral edge to the center of the brain is
            # ~ PA:10 mm, LR:7.5 mm, and IS:5 mm (see DOI: 10.1089/089771503770802853)
            # roll movement is most likely to occur, so set to 7.5 mm
            config.workflow.fd_radius = 7.5
            # block uploads for the moment; can be reversed before wider release
            config.execution.no_sub = True
