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
import os
import sys
from pathlib import Path
from time import strftime
from uuid import uuid4

try:
    # This option is only available with Python 3.8
    from importlib.metadata import version as get_version
except ImportError:
    from importlib_metadata import version as get_version

# Ignore annoying warnings
from mriqc._warnings import logging

__version__ = get_version("mriqc")
_pre_exec_env = dict(os.environ)

# Reduce numpy's vms by limiting OMP_NUM_THREADS
_default_omp_threads = int(os.getenv("OMP_NUM_THREADS", os.cpu_count()))

# Disable NiPype etelemetry always
_disable_et = bool(
    os.getenv("NO_ET") is not None or os.getenv("NIPYPE_NO_ET") is not None
)
os.environ["NIPYPE_NO_ET"] = "1"
os.environ["NO_ET"] = "1"

if not hasattr(sys, "_is_pytest_session"):
    sys._is_pytest_session = False  # Trick to avoid sklearn's FutureWarnings
# Disable all warnings in main and children processes only on production versions
if not any(
    (
        "+" in __version__,
        __version__.endswith(".dirty"),
        os.getenv("MRIQC_DEV", "0").lower() in ("1", "on", "true", "y", "yes"),
    )
):
    os.environ["PYTHONWARNINGS"] = "ignore"

logging.addLevelName(25, "IMPORTANT")  # Add a new level between INFO and WARNING
logging.addLevelName(15, "VERBOSE")  # Add a new level between INFO and DEBUG

DEFAULT_MEMORY_MIN_GB = 0.01
DSA_MESSAGE = """\
IMPORTANT: Anonymized quality metrics (IQMs) will be submitted to MRIQC's metrics \
repository. \
Submission of IQMs can be disabled using the ``--no-sub`` argument. \
Please visit https://mriqc.readthedocs.io/en/latest/dsa.html to revise MRIQC's \
Data Sharing Agreement."""

_exec_env = os.name
_docker_ver = None
# special variable set in the container
if os.getenv("IS_DOCKER_8395080871"):
    _exec_env = "singularity"
    _cgroup = Path("/proc/1/cgroup")
    if _cgroup.exists() and "docker" in _cgroup.read_text():
        _docker_ver = os.getenv("DOCKER_VERSION_8395080871")
        _exec_env = "docker"
    del _cgroup

_templateflow_home = Path(
    os.getenv(
        "TEMPLATEFLOW_HOME",
        os.path.join(os.getenv("HOME"), ".cache", "templateflow"),
    )
)

try:
    from psutil import virtual_memory

    _free_mem_at_start = round(virtual_memory().free / 1024**3, 1)
except Exception:
    _free_mem_at_start = None

_oc_limit = "n/a"
_oc_policy = "n/a"
try:
    # Memory policy may have a large effect on types of errors experienced
    _proc_oc_path = Path("/proc/sys/vm/overcommit_memory")
    if _proc_oc_path.exists():
        _oc_policy = {"0": "heuristic", "1": "always", "2": "never"}.get(
            _proc_oc_path.read_text().strip(), "unknown"
        )
        if _oc_policy != "never":
            _proc_oc_kbytes = Path("/proc/sys/vm/overcommit_kbytes")
            if _proc_oc_kbytes.exists():
                _oc_limit = _proc_oc_kbytes.read_text().strip()
            if (
                _oc_limit in ("0", "n/a")
                and Path("/proc/sys/vm/overcommit_ratio").exists()
            ):
                _oc_limit = "{}%".format(
                    Path("/proc/sys/vm/overcommit_ratio").read_text().strip()
                )
except Exception:
    pass

_memory_gb = None
try:
    if "linux" in sys.platform:
        with open("/proc/meminfo", "r") as f_in:
            _meminfo_lines = f_in.readlines()
            _mem_total_line = [line for line in _meminfo_lines if "MemTotal" in line][0]
            _mem_total = float(_mem_total_line.split()[1])
            _memory_gb = _mem_total / (1024.0**2)
    elif "darwin" in sys.platform:
        _mem_str = os.popen("sysctl hw.memsize").read().strip().split(" ")[-1]
        _memory_gb = float(_mem_str) / (1024.0**3)
except Exception:
    pass


class _Config:
    """An abstract class forbidding instantiation."""

    _paths = tuple()

    def __init__(self):
        """Avert instantiation."""
        raise RuntimeError("Configuration type is not instantiable.")

    @classmethod
    def load(cls, settings, init=True):
        """Store settings from a dictionary."""
        for k, v in settings.items():
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
    def get(cls):
        """Return defined settings."""
        out = {}
        for k, v in cls.__dict__.items():
            if k.startswith("_") or v is None:
                continue
            if callable(getattr(cls, k)):
                continue
            if k in cls._paths:
                v = str(v)
            out[k] = v
        return out


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

    cpu_count = os.cpu_count()
    """Number of available CPUs."""
    exec_docker_version = _docker_ver
    """Version of Docker Engine."""
    exec_env = _exec_env
    """A string representing the execution platform."""
    free_mem = _free_mem_at_start
    """Free memory at start."""
    overcommit_policy = _oc_policy
    """Linux's kernel virtual memory overcommit policy."""
    overcommit_limit = _oc_limit
    """Linux's kernel virtual memory overcommit limits."""
    nipype_version = get_version("nipype")
    """Nipype's current version."""
    templateflow_version = get_version("templateflow")
    """The TemplateFlow client version installed."""
    total_memory = _memory_gb
    """Total memory available, in GB."""
    version = __version__
    """*MRIQC*'s version."""
    _pre_mriqc = _pre_exec_env
    """Environment variables before MRIQC's execution."""


class nipype(_Config):
    """Nipype settings."""

    crashfile_format = "txt"
    """The file format for crashfiles, either text or pickle."""
    get_linked_libs = False
    """Run NiPype's tool to enlist linked libraries for every interface."""
    local_hash_check = True
    """Check if interface is cached locally before executing."""
    memory_gb = None
    """Estimation in GB of the RAM this workflow can allocate at any given time."""
    nprocs = os.cpu_count()
    """Number of processes (compute tasks) that can be run in parallel (multiprocessing only)."""
    omp_nthreads = _default_omp_threads
    """Number of CPUs a single process can access for multithreaded execution."""
    plugin = "MultiProc"
    """NiPype's execution plugin."""
    plugin_args = {
        "maxtasksperchild": 1,
        "raise_insufficient": False,
    }
    """Settings for NiPype's execution plugin."""
    remove_node_directories = False
    """Remove directories whose outputs have already been used up."""
    resource_monitor = False
    """Enable resource monitor."""
    stop_on_first_crash = True
    """Whether the workflow should stop or continue after the first error."""

    @classmethod
    def get_plugin(cls):
        """Format a dictionary for Nipype consumption."""
        out = {
            "plugin": cls.plugin,
            "plugin_args": cls.plugin_args,
        }
        if cls.plugin in ("MultiProc", "LegacyMultiProc"):
            out["plugin_args"]["n_procs"] = int(cls.nprocs)
            if cls.memory_gb:
                out["plugin_args"]["memory_gb"] = float(cls.memory_gb)
        return out

    @classmethod
    def init(cls):
        """Set NiPype configurations."""
        from nipype import config as ncfg

        # Nipype config (logs and execution)
        ncfg.update_config(
            {
                "execution": {
                    "crashdump_dir": str(execution.log_dir),
                    "crashfile_format": cls.crashfile_format,
                    "get_linked_libs": cls.get_linked_libs,
                    "stop_on_first_crash": cls.stop_on_first_crash,
                }
            }
        )


class execution(_Config):
    """Configure run-level settings."""

    ants_float = False
    """Use float number precision for ANTs computations."""
    bids_dir = None
    """An existing path to the dataset, which must be BIDS-compliant."""
    bids_database_dir = None
    """Path to the directory containing SQLite database indices for the input BIDS dataset."""
    bids_description_hash = None
    """Checksum (SHA256) of the ``dataset_description.json`` of the BIDS dataset."""
    cwd = os.getcwd()
    """Current working directory."""
    debug = False
    """Run in sloppy mode (meaning, suboptimal parameters that minimize run-time)."""
    dry_run = False
    """Just test, do not run."""
    dsname = "<unset>"
    """A dataset name used when generating files from the rating widget."""
    echo_id = None
    """Select a particular echo for multi-echo EPI datasets."""
    float32 = True
    """Cast the input data to float32 if it's represented with higher precision."""
    layout = None
    """A :py:class:`~bids.layout.BIDSLayout` object, see :py:func:`init`."""
    log_dir = None
    """The path to a directory that contains execution logs."""
    log_level = 25
    """Output verbosity."""
    modalities = None
    """Filter input dataset by MRI type."""
    no_sub = False
    """Turn off submission of anonymized quality metrics to Web API."""
    notrack = False
    """Disable the sharing of usage information with developers."""
    output_dir = None
    """Folder where derivatives will be stored."""
    participant_label = None
    """List of participant identifiers that are to be preprocessed."""
    pdb = False
    """Drop into PDB when exceptions are encountered."""
    reports_only = False
    """Only build the reports, based on the reportlets found in a cached working directory."""
    resource_monitor = False
    """Enable resource monitor."""
    run_id = None
    """Filter input dataset by run identifier."""
    run_uuid = "%s_%s" % (strftime("%Y%m%d-%H%M%S"), uuid4())
    """Unique identifier of this particular run."""
    session_id = None
    """Filter input dataset by session identifier."""
    task_id = None
    """Select a particular task from all available in the dataset."""
    templateflow_home = _templateflow_home
    """The root folder of the TemplateFlow client."""
    upload_strict = False
    """Workflow will crash if upload is not successful."""
    verbose_reports = False
    """Generate extended reports."""
    webapi_url = "https://mriqc.nimh.nih.gov/api/v1"
    """IP address where the MRIQC WebAPI is listening."""
    webapi_port = None
    """port where the MRIQC WebAPI is listening."""
    work_dir = Path("work").absolute()
    """Path to a working directory where intermediate results will be available."""
    write_graph = False
    """Write out the computational graph corresponding to the planned preprocessing."""

    _layout = None

    _paths = (
        "anat_derivatives",
        "bids_dir",
        "bids_database_dir",
        "fs_license_file",
        "fs_subjects_dir",
        "layout",
        "log_dir",
        "output_dir",
        "templateflow_home",
        "work_dir",
    )

    @classmethod
    def init(cls):
        """Create a new BIDS Layout accessible with :attr:`~execution.layout`."""
        if cls._layout is None:
            import re
            from bids.layout.index import BIDSLayoutIndexer
            from bids.layout import BIDSLayout

            ignore_paths = [
                # Ignore folders at the top if they don't start with /sub-<label>/
                re.compile(r"^(?!/sub-[a-zA-Z0-9]+)"),
                # Ignore all modality subfolders, except for func/ or anat/
                re.compile(
                    r"^/sub-[a-zA-Z0-9]+(/ses-[a-zA-Z0-9]+)?/(?!func|anat)"
                ),
                # Ignore all files, except for the supported modalities
                re.compile(r"^.+(?<!(_T1w|_T2w|bold))\.(json|nii|nii\.gz)$"),
            ]

            if cls.participant_label:
                # If we know participant labels, ignore all other
                ignore_paths[0] = re.compile(
                    r"^(?!/sub-("
                    + "|".join(cls.participant_label)
                    + r"))"
                )

            _db_path = cls.bids_database_dir or (
                cls.work_dir / cls.run_uuid / "bids_db"
            )
            _db_path.mkdir(exist_ok=True, parents=True)

            # Recommended after PyBIDS 12.1
            _indexer = BIDSLayoutIndexer(
                validate=False,
                ignore=ignore_paths,
            )
            cls._layout = BIDSLayout(
                str(cls.bids_dir),
                database_path=_db_path,
                reset_database=cls.bids_database_dir is None,
                indexer=_indexer,
            )
            cls.bids_database_dir = _db_path

        cls.layout = cls._layout


# These variables are not necessary anymore
del _exec_env
del _templateflow_home
del _free_mem_at_start
del _oc_limit
del _oc_policy


class workflow(_Config):
    """Configure the particular execution graph of this workflow."""

    analysis_level = ["participant"]
    """Level of analysis."""
    biggest_file_gb = 1
    """Size of largest file in GB."""
    deoblique = False
    """Deoblique the functional scans during head motion correction preprocessing."""
    despike = False
    """Despike the functional scans during head motion correction preprocessing."""
    fd_thres = 0.2
    """Threshold on Framewise Displacement estimates to detect outliers."""
    fd_radius = 50
    """Radius in mm. of the sphere for the FD calculation."""
    fft_spikes_detector = False
    """Turn on FFT based spike detector (slow)."""
    headmask = "BET"
    """Use FSL BET in :py:func:`~mriqc.workflows.anatomical.headmsk_wf`."""
    ica = False
    """Run ICA on the raw data and include the components in the individual reports."""
    inputs = None
    """List of files to be processed with MRIQC."""
    species = "human"
    """Subject species to choose most appropriate template"""
    template_id = "MNI152NLin2009cAsym"
    """TemplateFlow ID of template used for the anatomical processing."""


class loggers:
    """Keep loggers easily accessible (see :py:func:`init`)."""

    _fmt = "%(asctime)s,%(msecs)d %(name)-2s " "%(levelname)-2s:\n\t %(message)s"
    _datefmt = "%y%m%d-%H:%M:%S"
    _init = False

    default = logging.getLogger()
    """The root logger."""
    cli = logging.getLogger("cli")
    """Command-line interface logging."""
    workflow = None
    """NiPype's workflow logger."""
    interface = None
    """NiPype's interface logger."""
    utils = None
    """NiPype's utils logger."""

    @classmethod
    def init(cls):
        """
        Set the log level, initialize all loggers into :py:class:`loggers`.

            * Add new logger levels (25: IMPORTANT, and 15: VERBOSE).
            * Add a new sub-logger (``cli``).
            * Logger configuration.

        """
        if not cls._init:
            from nipype import logging as nlogging
            from nipype import config as ncfg

            cls.workflow = nlogging.getLogger("nipype.workflow")
            cls.interface = nlogging.getLogger("nipype.interface")
            cls.utils = nlogging.getLogger("nipype.utils")

            if not len(cls.cli.handlers):
                _handler = logging.StreamHandler(stream=sys.stdout)
                _handler.setFormatter(
                    logging.Formatter(fmt=cls._fmt, datefmt=cls._datefmt)
                )
                cls.cli.addHandler(_handler)

            ncfg.update_config(
                {
                    "logging": {
                        "log_directory": str(execution.log_dir),
                        "log_to_file": True,
                    },
                }
            )
            cls._init = True

        cls.default.setLevel(execution.log_level)
        cls.cli.setLevel(execution.log_level)
        cls.interface.setLevel(execution.log_level)
        cls.workflow.setLevel(execution.log_level)
        cls.utils.setLevel(execution.log_level)

    @classmethod
    def getLogger(cls, name):
        """Create a new logger."""
        retval = getattr(cls, name)
        if retval is None:
            setattr(cls, name, logging.getLogger(name))
            _handler = logging.StreamHandler(stream=sys.stdout)
            _handler.setFormatter(logging.Formatter(fmt=cls._fmt, datefmt=cls._datefmt))
            retval.addHandler(_handler)
            retval.setLevel(execution.log_level)
        return retval


def from_dict(settings):
    """Read settings from a flat dictionary."""
    execution.load(settings)
    workflow.load(settings)
    nipype.load(settings, init=False)


def load(filename):
    """Load settings from file."""
    from toml import loads

    filename = Path(filename)
    settings = loads(filename.read_text())
    for sectionname, configs in settings.items():
        if sectionname != "environment":
            section = getattr(sys.modules[__name__], sectionname)
            section.load(configs)


def get(flat=False):
    """Get config as a dict."""
    settings = {
        "environment": environment.get(),
        "execution": execution.get(),
        "workflow": workflow.get(),
        "nipype": nipype.get(),
    }
    if not flat:
        return settings

    return {
        ".".join((section, k)): v
        for section, configs in settings.items()
        for k, v in configs.items()
    }


def dumps():
    """Format config into toml."""
    from toml import dumps

    return dumps(get())


def to_filename(filename):
    """Write settings to file."""
    filename = Path(filename)
    filename.parent.mkdir(exist_ok=True, parents=True)
    filename.write_text(dumps())


def _process_initializer(cwd, omp_nthreads):
    """Initialize the environment of the child process."""
    os.chdir(cwd)
    os.environ["NIPYPE_NO_ET"] = "1"
    os.environ["OMP_NUM_THREADS"] = f"{omp_nthreads}"


def restore_env():
    """Restore the original environment."""

    for k in os.environ.keys():
        del os.environ[k]

    for k, v in environment._pre_mriqc.items():
        os.environ[k] = v
