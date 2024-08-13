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
r"""
A Python module to maintain unique, run-wide *MRIQC* settings.

This module implements the memory structures to keep a consistent, singleton config.
Settings are passed across processes via filesystem, and a copy of the settings for
each run and subject is left under
``<output_dir>/sub-<participant_id>/log/<run_unique_id>/mriqc.toml``.
Settings are stored using :abbr:`ToML (Tom's Markup Language)`.
The module has a :py:func:`~mriqc.config.to_filename` function to allow writing out
the settings to hard disk in *ToML* format, which looks like:

.. literalinclude:: ../mriqc/data/config-example.toml
   :language: toml
   :name: mriqc.toml
   :caption: **Example file representation of MRIQC settings**.

This config file is used to pass the settings across processes,
using the :py:func:`~mriqc.config.load` function.

Configuration sections
----------------------
.. autoclass:: environment
   :members:
.. autoclass:: execution
   :members:
.. autoclass:: workflow
   :members:
.. autoclass:: nipype
   :members:

Usage
-----
A config file is used to pass settings and collect information as the execution
graph is built across processes.

.. code-block:: Python

    from mriqc import config
    config_file = mktemp(dir=config.execution.work_dir, prefix='.mriqc.', suffix='.toml')
    config.to_filename(config_file)
    # Call build_workflow(config_file, retval) in a subprocess
    with Manager() as mgr:
        from .workflow import build_workflow
        retval = mgr.dict()
        p = Process(target=build_workflow, args=(str(config_file), retval))
        p.start()
        p.join()
    config.load(config_file)
    # Access configs from any code section as:
    value = config.section.setting

Logging
-------
.. autoclass:: loggers
   :members:

Other responsibilities
----------------------
The :py:mod:`config` is responsible for other conveniency actions.

  * Switching Python's :obj:`multiprocessing` to *forkserver* mode.
  * Set up a filter for warnings as early as possible.
  * Automated I/O magic operations. Some conversions need to happen in the
    store/load processes (e.g., from/to :obj:`~pathlib.Path` \<-\> :obj:`str`,
    :py:class:`~bids.layout.BIDSLayout`, etc.)

"""

from __future__ import annotations

import os
import pickle
import sys
from contextlib import suppress
from pathlib import Path
from time import strftime
from typing import TYPE_CHECKING, Any, Iterable
from uuid import uuid4

try:
    # This option is only available with Python 3.8
    from importlib.metadata import version as get_version
except ImportError:
    from importlib_metadata import version as get_version

# Ignore annoying warnings
from mriqc._warnings import logging

if TYPE_CHECKING:
    from re import Pattern

    from bids.layout import BIDSLayout

__version__: str = get_version('mriqc')
_pre_exec_env: dict[str, str] = dict(os.environ)

# Reduce numpy's vms by limiting OMP_NUM_THREADS
_default_omp_threads: int = int(os.getenv('OMP_NUM_THREADS', os.cpu_count()))

# Disable NiPype etelemetry always
_disable_et: bool = bool(os.getenv('NO_ET') is not None or os.getenv('NIPYPE_NO_ET') is not None)
os.environ['NIPYPE_NO_ET'] = '1'
os.environ['NO_ET'] = '1'

_mriqc_dev: bool = os.getenv('MRIQC_DEV', '0').lower() in ('1', 'on', 'true', 'y', 'yes')

if not hasattr(sys, '_is_pytest_session'):
    sys._is_pytest_session = False  # Trick to avoid sklearn's FutureWarnings
# Disable all warnings in main and children processes only on production versions
if not any(
    (
        '+' in __version__,
        __version__.endswith('.dirty'),
        _mriqc_dev,
    )
):
    os.environ['PYTHONWARNINGS'] = 'ignore'


SUPPORTED_SUFFIXES: tuple[str, ...] = ('T1w', 'T2w', 'bold', 'dwi')

DEFAULT_MEMORY_MIN_GB: float = 0.01
DSA_MESSAGE: str = """\
IMPORTANT: Anonymized quality metrics (IQMs) will be submitted to MRIQC's metrics \
repository. \
Submission of IQMs can be disabled using the ``--no-sub`` argument. \
Please visit https://mriqc.readthedocs.io/en/latest/dsa.html to revise MRIQC's \
Data Sharing Agreement."""

_exec_env: str = os.name
_docker_ver: str | None = None
# special variable set in the container
if os.getenv('IS_DOCKER_8395080871'):
    _exec_env: str = 'singularity'
    _cgroup: Path = Path('/proc/1/cgroup')
    if _cgroup.exists() and 'docker' in _cgroup.read_text():
        _docker_ver = os.getenv('DOCKER_VERSION_8395080871')
        _exec_env = 'docker'
    del _cgroup

_templateflow_home: Path = Path(
    os.getenv(
        'TEMPLATEFLOW_HOME',
        os.path.join(os.getenv('HOME'), '.cache', 'templateflow'),
    )
)

_free_mem_at_start: int | None = None
with suppress(Exception):
    from psutil import virtual_memory

    _free_mem_at_start: int = round(virtual_memory().free / 1024**3, 1)

_oc_limit: str = 'n/a'
_oc_policy: str = 'n/a'
with suppress(Exception):
    # Memory policy may have a large effect on types of errors experienced
    _proc_oc_path: Path = Path('/proc/sys/vm/overcommit_memory')
    if _proc_oc_path.exists():
        _oc_policy: str = {'0': 'heuristic', '1': 'always', '2': 'never'}.get(
            _proc_oc_path.read_text().strip(), 'unknown'
        )
        if _oc_policy != 'never':
            _proc_oc_kbytes: Path = Path('/proc/sys/vm/overcommit_kbytes')
            if _proc_oc_kbytes.exists():
                _oc_limit: str = _proc_oc_kbytes.read_text().strip()
            if _oc_limit in ('0', 'n/a') and Path('/proc/sys/vm/overcommit_ratio').exists():
                _oc_limit: str = '{}%'.format(
                    Path('/proc/sys/vm/overcommit_ratio').read_text().strip()
                )

_memory_gb: float | None = None

if 'linux' in sys.platform:
    with suppress(Exception):
        with open('/proc/meminfo') as f_in:
            _meminfo_lines: list[str] = f_in.readlines()
            _mem_total_line: str = [line for line in _meminfo_lines if 'MemTotal' in line][0]
            _mem_total: float = float(_mem_total_line.split()[1])
            _memory_gb = _mem_total / (1024.0**2)
elif 'darwin' in sys.platform:
    from shutil import which
    from subprocess import check_output

    if _cmd := which('sysctl'):
        with suppress(Exception):
            _mem_str: str = check_output([_cmd, 'hw.memsize']).decode().strip().split(' ')[-1]
            _memory_gb: float = float(_mem_str) / (1024.0**3)

# Check for FreeSurfer's SynthStrip model
_fs_home: str | None = os.getenv('FREESURFER_HOME', None)
_default_model_path: Path | None = (
    Path(_fs_home) / 'models' / 'synthstrip.1.pt' if _fs_home else None
)

if _fs_home and not _default_model_path.exists():
    _default_model_path = None

# Override the unique ID if MRIQC_DEV is set
_run_uuid: str = (
    '{}_{}'.format(strftime('%Y%m%d-%H%M%S'), uuid4())
    if not _mriqc_dev
    else '18480913-163000_PhineasG-ageh-adhi-sacc-ident9b1ab0f'
)

_mriqc_cache: Path = Path(
    os.getenv('MRIQC_CACHE_PATH', str(Path.home() / '.cache' / 'mriqc'))
).absolute()


class _Config:
    """An abstract class forbidding instantiation."""

    _paths: tuple[str, ...] = ()
    _hidden: tuple[str, ...] = ()

    def __init__(self):
        """Avert instantiation."""
        raise RuntimeError('Configuration type is not instantiable.')

    @classmethod
    def load(cls, sections, init=True):
        """Store settings from a dictionary."""
        for k, v in sections.items():
            if v is None:
                continue
            if k in cls._paths:
                setattr(cls, k, Path(v).absolute())
                continue
            if hasattr(cls, k):
                setattr(cls, k, v)

        if init:
            try:
                cls.init()
            except AttributeError:
                pass

    @classmethod
    def get(cls) -> dict[str, Any]:
        """Return defined settings."""
        out: dict[str, str] = {}
        for k, v in cls.__dict__.items():
            if k.startswith('_') or v is None:
                continue
            if k in cls._hidden:
                continue
            if callable(getattr(cls, k)):
                continue
            if k in cls._paths:
                v = str(v)
            out[k] = v
        return out


class settings(_Config):
    """Settings of this config module."""

    file_path: Path | None = None
    """Path to this configuration file."""
    start_time: float | None = None
    """A :obj:`~time.time` timestamp at the time the workflow is started."""

    _paths: tuple[str, ...] = ('file_path',)


class environment(_Config):
    """
    Read-only options regarding the platform and environment.

    Crawls runtime descriptive settings (e.g., default FreeSurfer license,
    execution environment, nipype and *MRIQC* versions, etc.).
    The ``environment`` section is not loaded in from file,
    only written out when settings are exported.
    This config section is useful when reporting issues,
    and these variables are tracked whenever the user does not
    opt-out using the ``--notrack`` argument.

    """

    cache_path: Path = _mriqc_cache
    """Path to the location of the cache directory."""
    cpu_count: int = os.cpu_count()
    """Number of available CPUs."""
    exec_docker_version: str | None = _docker_ver
    """Version of Docker Engine."""
    exec_env: str = _exec_env
    """A string representing the execution platform."""
    free_mem: int | None = _free_mem_at_start
    """Free memory at start."""
    freesurfer_home: str | None = _fs_home
    """Path to the *FreeSurfer* installation (from ``FREESURFER_HOME`` environment variable)."""
    overcommit_policy: str = _oc_policy
    """Linux's kernel virtual memory overcommit policy."""
    overcommit_limit: str = _oc_limit
    """Linux's kernel virtual memory overcommit limits."""
    nipype_version: str = get_version('nipype')
    """Nipype's current version."""
    synthstrip_path: Path | None = _default_model_path
    """Path to *SynthStrip*'s model weights (requires *FreeSurfer*)."""
    templateflow_version: str = get_version('templateflow')
    """The TemplateFlow client version installed."""
    total_memory: float | None = _memory_gb
    """Total memory available, in GB."""
    version: str = __version__
    """*MRIQC*'s version."""
    _pre_mriqc: dict[str, str] = _pre_exec_env
    """Environment variables before MRIQC's execution."""


class nipype(_Config):
    """Nipype settings."""

    crashfile_format: str = 'txt'
    """The file format for crashfiles, either text or pickle."""
    get_linked_libs: bool = False
    """Run NiPype's tool to enlist linked libraries for every interface."""
    local_hash_check: bool = True
    """Check if interface is cached locally before executing."""
    memory_gb: float | str | None = None
    """Estimation in GB of the RAM this workflow can allocate at any given time."""
    nprocs: int = os.cpu_count()
    """Number of processes (compute tasks) that can be run in parallel (multiprocessing only)."""
    omp_nthreads: int = _default_omp_threads
    """Number of CPUs a single process can access for multithreaded execution."""
    plugin: str = 'MultiProc'
    """NiPype's execution plugin."""
    plugin_args: dict[str, Any] = {
        'maxtasksperchild': 1,
        'raise_insufficient': False,
    }
    """Settings for NiPype's execution plugin."""
    remove_node_directories: bool = False
    """Remove directories whose outputs have already been used up."""
    resource_monitor: bool = False
    """Enable resource monitor."""
    stop_on_first_crash: bool = True
    """Whether the workflow should stop or continue after the first error."""

    @classmethod
    def get_plugin(cls) -> dict[str, Any]:
        """Format a dictionary for Nipype consumption."""
        out: dict[str, Any] = {
            'plugin': cls.plugin,
            'plugin_args': cls.plugin_args,
        }
        if cls.plugin in ('MultiProc', 'LegacyMultiProc'):
            out['plugin_args']['n_procs'] = int(cls.nprocs)
            if cls.memory_gb:
                out['plugin_args']['memory_gb'] = float(cls.memory_gb)
        return out

    @classmethod
    def init(cls) -> None:
        """Set NiPype configurations."""
        from nipype import config as ncfg

        # Nipype config (logs and execution)
        ncfg.update_config(
            {
                'execution': {
                    'crashdump_dir': str(execution.log_dir),
                    'crashfile_format': cls.crashfile_format,
                    'get_linked_libs': cls.get_linked_libs,
                    'stop_on_first_crash': cls.stop_on_first_crash,
                }
            }
        )


class execution(_Config):
    """Configure run-level settings."""

    ants_float: bool = False
    """Use float number precision for ANTs computations."""
    bids_dir: str | os.PathLike | None = None
    """An existing path to the dataset, which must be BIDS-compliant."""
    bids_dir_datalad: bool = False
    """Whether the input directory seems under DataLad's version control."""
    bids_database_dir: str | os.PathLike | None = None
    """Path to the directory containing SQLite database indices for the input BIDS dataset."""
    bids_database_wipe: bool = False
    """Wipe out previously existing BIDS indexing caches, forcing re-indexing."""
    bids_description_hash: str | None = None
    """Checksum (SHA256) of the ``dataset_description.json`` of the BIDS dataset."""
    bids_filters: dict | None = None
    """A dictionary describing custom BIDS input filter using PyBIDS."""
    cwd: str = os.getcwd()
    """Current working directory."""
    datalad_get: bool = True
    """
    If :obj:`~mriqc.config.execution.bids_dir` is a DataLad dataset, a ``datalad get`` command
    will be executed to ensure all remote files have been fetched.
    This behavior can be disabled with the command line option ``--no-datalad-get``.
    """
    debug: bool = False
    """Run in sloppy mode (meaning, suboptimal parameters that minimize run-time)."""
    dry_run: bool = False
    """Just test, do not run."""
    dsname: str = '<unset>'
    """A dataset name used when generating files from the rating widget."""
    echo_id: str | None = None
    """Select a particular echo for multi-echo EPI datasets."""
    float32: bool = True
    """Cast the input data to float32 if it's represented with higher precision."""
    layout: BIDSLayout | None = None
    """A :py:class:`~bids.layout.BIDSLayout` object, see :py:func:`init`."""
    log_dir: str | os.PathLike | None = None
    """The path to a directory that contains execution logs."""
    log_level: int | str = 25
    """Output verbosity."""
    modalities: Iterable[str] = None
    """Filter input dataset by MRI type."""
    no_sub: bool = False
    """Turn off submission of anonymized quality metrics to Web API."""
    notrack: bool = False
    """Disable the sharing of usage information with developers."""
    output_dir: str | os.PathLike | None = None
    """Folder where derivatives will be stored."""
    participant_label: Iterable[str] | None = None
    """List of participant identifiers that are to be preprocessed."""
    pdb: bool = False
    """Drop into PDB when exceptions are encountered."""
    reports_only: bool = False
    """Only build the reports, based on the reportlets found in a cached working directory."""
    resource_monitor: bool = False
    """Enable resource monitor."""
    run_id: str | None = None
    """Filter input dataset by run identifier."""
    run_uuid: str = _run_uuid
    """Unique identifier of this particular run."""
    session_id: str | None = None
    """Filter input dataset by session identifier."""
    task_id: str | None = None
    """Select a particular task from all available in the dataset."""
    templateflow_home: str | os.PathLike | None = _templateflow_home
    """The root folder of the TemplateFlow client."""
    upload_strict: bool = False
    """Workflow will crash if upload is not successful."""
    verbose_reports: bool = False
    """Generate extended reports."""
    webapi_token: str = '<secret_token>'
    """Authorization token for the WebAPI service."""
    webapi_url: str = 'https://mriqc.nimh.nih.gov:443/api/v1'
    """IP address where the MRIQC WebAPI is listening."""
    work_dir: Path = Path('work').absolute()
    """Path to a working directory where intermediate results will be available."""
    write_graph: bool = False
    """Write out the computational graph corresponding to the planned preprocessing."""

    _layout: BIDSLayout | None = None

    _paths: tuple[str, ...] = (
        'anat_derivatives',
        'bids_dir',
        'bids_database_dir',
        'fs_license_file',
        'fs_subjects_dir',
        'log_dir',
        'output_dir',
        'templateflow_home',
        'work_dir',
    )

    _hidden: tuple[str, ...] = ('webapi_token',)

    @classmethod
    def init(cls) -> None:
        """Create a new BIDS Layout accessible with :attr:`~execution.layout`."""

        if cls.bids_filters is None:
            cls.bids_filters = {}

        # Process --run-id if the argument was provided
        if cls.run_id:
            for mod in cls.modalities:
                cls.bids_filters.setdefault(mod.lower(), {})['run'] = cls.run_id

        if cls._layout is None:
            import re

            from bids.layout import BIDSLayout
            from bids.layout.index import BIDSLayoutIndexer

            ignore_paths: list[Pattern] = [
                # Ignore folders at the top if they don't start with /sub-<label>/
                re.compile(r'^(?!/sub-[a-zA-Z0-9]+)'),
                # Ignore all modality subfolders, except for func/ or anat/
                re.compile(
                    r'^/sub-[a-zA-Z0-9]+(/ses-[a-zA-Z0-9]+)?/'
                    r'(beh|fmap|pet|perf|meg|eeg|ieeg|micr|nirs)'
                ),
                # Ignore all files, except for the supported modalities
                re.compile(r'^.+(?<!(_T1w|_T2w|bold|_dwi))\.(json|nii|nii\.gz)$'),
            ]

            if cls.participant_label:
                # If we know participant labels, ignore all other
                ignore_paths[0] = re.compile(
                    r'^(?!/sub-(' + '|'.join(cls.participant_label) + '))'
                )

            # Recommended after PyBIDS 12.1
            _indexer: BIDSLayoutIndexer = BIDSLayoutIndexer(
                validate=False,
                ignore=ignore_paths,
            )

            # Initialize database in a multiprocessing-safe manner
            _db_path: Path = (
                cls.work_dir if cls.participant_label else cls.output_dir
            ) / f'.bids_db-{cls.run_uuid}'

            if cls.bids_database_dir is None:
                cls.bids_database_dir = (
                    cls.output_dir / '.bids_db' if not cls.participant_label else _db_path
                )

            if cls.bids_database_wipe or not cls.bids_database_dir.exists():
                _db_path.mkdir(exist_ok=True, parents=True)

                cls._layout = BIDSLayout(
                    str(cls.bids_dir),
                    database_path=_db_path,
                    indexer=_indexer,
                )

                if _db_path != cls.bids_database_dir:
                    _db_path.replace(cls.bids_database_dir.absolute())

            cls._layout = BIDSLayout(
                str(cls.bids_dir),
                database_path=cls.bids_database_dir,
                indexer=_indexer,
            )

            # Rewrite __repr__ to avoid the layout query and build the summary
            # For a smallish dataset this takes one minute each time.
            # See https://github.com/nipreps/mriqc/issues/1239
            cls._layout.__class__.__repr__ = lambda x: f'BIDS Layout: {cls.bids_dir}'

        cls.layout: BIDSLayout = cls._layout


# These variables are not necessary anymore
del _exec_env
del _templateflow_home
del _free_mem_at_start
del _oc_limit
del _oc_policy


class workflow(_Config):
    """Configure the particular execution graph of this workflow."""

    analysis_level: list[str] = ['participant']
    """Level of analysis."""
    biggest_file_gb: dict[int] = 1
    """Dictionary holding the size of largest file in GB (per modality)."""
    deoblique: bool = False
    """Deoblique the functional scans during head motion correction preprocessing."""
    despike: bool = False
    """Despike the functional scans during head motion correction preprocessing."""
    fd_thres: float = 0.2
    """Threshold on Framewise Displacement estimates to detect outliers."""
    fd_radius: int = 50
    """Radius in mm. of the sphere for the FD calculation."""
    fft_spikes_detector: bool = False
    """Turn on FFT based spike detector (slow)."""
    inputs: list[str | os.PathLike] | None = None
    """List of files to be processed with MRIQC."""
    inputs_entities: dict[list[dict]]
    """List of entities corresponding to inputs."""
    inputs_metadata: dict[list[dict | list[dict]]] | None = None
    """List of metadata corresponding to inputs."""
    inputs_path: Path | None = None
    """Path to a pickle file with the input paths and metadata."""
    min_len_dwi: int = 7
    """
    Minimum DWI length to be considered a "processable" dataset
    (default: 7, assuming one low-b and six gradients for diffusion tensor imaging).
    """
    min_len_bold: int = 5
    """Minimum BOLD length to be considered a "processable" dataset."""
    species: str = 'human'
    """Subject species to choose most appropriate template"""
    template_id: str = 'MNI152NLin2009cAsym'
    """TemplateFlow ID of template used for the anatomical processing."""

    _hidden: tuple[str, ...] = ('inputs', 'inputs_entities', 'inputs_metadata')

    @classmethod
    def init(cls) -> None:
        if cls.inputs_path is None:
            cls.inputs_path = execution.work_dir / f'inputs-{execution.run_uuid}.pkl'

        if cls.inputs_path.exists():
            with open(cls.inputs_path, 'rb') as handle:
                _inputs = pickle.load(handle)

                cls.inputs = _inputs['paths']
                cls.inputs_metadata = _inputs['metadata']
                cls.inputs_entities = _inputs['entities']


class loggers:
    """Keep loggers easily accessible (see :py:func:`init`)."""

    _datefmt: str = '%y%m%d %H:%M:%S'
    _init: bool = False
    _other_loggers: tuple[str, ...] = ('py.warnings', 'numexpr.utils', 'datalad')

    default: logging.Logger = logging.getLogger()
    """The root logger."""
    cli: logging.Logger = logging.getLogger('mriqc')
    """Command-line interface logging."""
    workflow: logging.Logger | None = None
    """NiPype's workflow logger."""
    interface: logging.Logger | None = None
    """NiPype's interface logger."""
    utils: logging.Logger | None = None
    """NiPype's utils logger."""

    @classmethod
    def init(cls) -> None:
        """
        Set the log level, initialize all loggers into :py:class:`loggers`.

            * Add new logger levels (25: IMPORTANT, and 15: VERBOSE).
            * Add a new sub-logger (``cli``).
            * Logger configuration.

        """
        if not cls._init:
            from nipype import config as ncfg
            from nipype import logging as nlogging

            cls.workflow = nlogging.getLogger('nipype.workflow')
            cls.interface = nlogging.getLogger('nipype.interface')
            cls.utils = nlogging.getLogger('nipype.utils')

            cls.workflow.handlers.clear()
            cls.interface.handlers.clear()
            cls.utils.handlers.clear()

            ncfg.update_config(
                {
                    'logging': {
                        'log_directory': str(execution.log_dir),
                        'log_to_file': True,
                    },
                }
            )

            for name in cls._other_loggers:
                _thislogger: logging.Logger = logging.getLogger(name)
                _thislogger.setLevel(logging.WARNING)

            cls._init = True

        cls.default.setLevel(execution.log_level)
        cls.cli.setLevel(execution.log_level)
        cls.interface.setLevel(execution.log_level)
        cls.workflow.setLevel(execution.log_level)
        cls.utils.setLevel(execution.log_level)

    @classmethod
    def getLogger(cls, name) -> logging.Logger:
        """Create a new logger."""
        retval: logging.Logger | None = getattr(cls, name)
        if retval is None:
            setattr(cls, name, logging.getLogger(name))
            retval.setLevel(execution.log_level)
        return retval


def from_dict(sections: dict) -> None:
    """Read settings from a flat dictionary."""
    execution.load(sections)
    workflow.load(sections)
    nipype.load(sections, init=False)


def load(filename: str | os.PathLike) -> None:
    """Load settings from file."""
    try:
        from tomllib import loads
    except ModuleNotFoundError:  # Python < 3.11
        from tomli import loads

    filename: Path = Path(filename)
    sections: dict[str, Any] = loads(filename.read_text())
    for sectionname, configs in sections.items():
        if sectionname != 'environment':
            section = getattr(sys.modules[__name__], sectionname)
            section.load(configs)

    if settings.file_path is None:
        settings.file_path = filename

    loggers.cli.debug(f'Loaded MRIQC config file: {settings.file_path}.')


def get(flat: bool = False) -> dict[str, dict[str, Any]]:
    """Get config as a dict."""
    sections = {
        'environment': environment.get(),
        'execution': execution.get(),
        'workflow': workflow.get(),
        'nipype': nipype.get(),
        'settings': settings.get(),
    }
    if not flat:
        return sections

    return {
        '.'.join((section, k)): v
        for section, configs in sections.items()
        for k, v in configs.items()
    }


def dumps() -> str:
    """Format config into toml."""
    from toml import dumps

    return dumps(get())


def to_filename(
    filename: str | os.PathLike | None = None,
    store_inputs: bool = True,
) -> Path:
    """Write settings to file."""

    if filename:
        settings.file_path = Path(filename)
    elif settings.file_path is None:
        settings.file_path = execution.work_dir / f'config-{execution.run_uuid}.toml'

    settings.file_path.parent.mkdir(exist_ok=True, parents=True)
    settings.file_path.write_text(dumps())
    loggers.cli.debug(f'Saved MRIQC config file: {settings.file_path}.')

    if store_inputs:
        if workflow.inputs_path is None:
            workflow.inputs_path = execution.work_dir / f'inputs-{execution.run_uuid}.pkl'

        # Pickle inputs
        with open(workflow.inputs_path, 'wb') as handle:
            inputs_dict = {
                'paths': workflow.inputs,
                'metadata': workflow.inputs_metadata,
                'entities': workflow.inputs_entities,
            }
            pickle.dump(inputs_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

        loggers.cli.debug(f'Saved MRIQC inputs file: {workflow.inputs_path}.')
    return settings.file_path


def _process_initializer(config_file: Path) -> None:
    """Initialize the environment of the child process."""
    from mriqc import config

    # Disable eTelemetry
    os.environ['NIPYPE_NO_ET'] = '1'
    os.environ['NO_ET'] = '1'

    # Load config
    config.load(config_file)

    # Initialize nipype config
    config.nipype.init()

    # Make sure loggers are started
    config.loggers.init()

    # Change working directory according to the config
    os.chdir(config.execution.cwd)

    # Set the maximal number of threads per process
    os.environ['OMP_NUM_THREADS'] = f'{config.nipype.omp_nthreads}'
    os.environ['NUMEXPR_MAX_THREADS'] = f'{config.nipype.omp_nthreads}'


def restore_env() -> None:
    """Restore the original environment."""

    for k in os.environ.keys():
        del os.environ[k]

    for k, v in environment._pre_mriqc.items():
        os.environ[k] = v
