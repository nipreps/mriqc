# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# Copyright 2022 The NiPreps Developers <nipreps@gmail.com>
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
"""Instrumentation to profile resource utilization."""
import warnings
from time import time
from datetime import datetime
from tempfile import mkstemp
from pathlib import Path
import threading
from contextlib import suppress
import psutil

_MB = 1024.0**2
SAMPLE_ATTRS = (
    "pid",
    "name",
    # "cmdline",
    "cpu_num",
    "cpu_percent",
    "memory_info",
    "num_threads",
    "num_fds",
)


def sample(
    pid=None,
    recursive=True,
    attrs=SAMPLE_ATTRS,
):
    """
    Probe process tree and snapshot current resource utilization.

    Parameters
    ----------
    pid : :obj:`int` or :obj:`None`
        The process ID that must be sampled. If ``None`` then it samples the
        current process from which ``sample()`` has been called.
    recursive : :obj:`bool`
        Whether the sampler should descend and explore the whole process tree.
    attrs : :obj:`iterable` of :obj:`str`
        A list of :obj:`psutil.Process` attribute names that will be retrieved when
        sampling.

    """

    proc_list = [psutil.Process(pid)]
    if proc_list and recursive:
        with suppress(psutil.NoSuchProcess):
            proc_list += proc_list[0].children(recursive=True)

    proc_info = []
    for process in proc_list:
        with suppress(psutil.NoSuchProcess):
            proc_info.append(process.as_dict(attrs=attrs))

    return proc_info


def sample2file(pid=None, recursive=True, timestamp=None, fd=None, flush=True):
    if fd is None:
        return

    print(
        "\n".join(
            [
                "\t".join(parse_sample(s, timestamp=timestamp))
                for s in sample(pid=pid, recursive=recursive)
            ]
        ),
        file=fd,
    )
    if flush:
        fd.flush()


def parse_sample(datapoint, timestamp=None, attrs=SAMPLE_ATTRS):
    """Convert a sample dictionary into a list of string values."""
    retval = [f"{timestamp or time()}"]

    for attr in attrs:
        value = datapoint.get(attr, None)
        if value is None:
            continue

        if attr == "cmdline":
            value = " ".join(value).replace("'", "\\'").replace('"', '\\"')
            value = [f"'{value}'"]
        elif attr == "memory_info":
            value = [f"{value.rss / _MB}", f"{value.vms / _MB}"]
        else:
            value = [f"{value}"]

        retval += value

    return retval


class ResourceRecorder(threading.Thread):
    """Attach a ``Thread`` to sample a specific PID with a certain frequence."""

    def __init__(self, pid=None, freq=0.2, log_file=None):
        """Initialize a resource recorder."""
        threading.Thread.__init__(self)

        self._freq = max(freq, 0.01)
        """Frequency (seconds) with which the probe must sample."""
        self._pid = pid
        """The process to be sampled."""

        _log_file = (
            Path(log_file)
            if log_file
            else Path(mkstemp(prefix="prof-", suffix=".tsv")[1])
        )
        _log_file.parent.mkdir(parents=True, exist_ok=True)

        # Open file and write headers (comment trace + header row)
        self._logfile = _log_file.absolute().open("w")

        _header = [
            datetime.now().strftime(
                "# MRIQC Resource recorder started (%Y/%m/%d; %H:%M:%S)"
            ),
            "\t".join(("timestamp", *SAMPLE_ATTRS)).replace(
                "memory_info", "mem_rss_mb\tmem_vsm_mb"
            ),
        ]
        print("\n".join(_header), file=self._logfile)
        sample2file(self._pid, fd=self._logfile)

        # Start thread
        self._event = threading.Event()
        self.start()

    def excepthook(self, args):
        print(
            "\n".join(
                [
                    datetime.now().strftime(
                        "# MRIQC Resource recorder stopped with error (%Y/%m/%d; %H:%M:%S)"
                    )
                ]
            ),
            file=self._logfile,
        )
        self._logfile.flush()
        self._logfile.close()

        if not self._event.is_set():
            self._event.set()

        tb = "\n    ".join(args.exc_traceback)
        warnings.warning(f"""ResourceRecorder errored.
    {args.exc_type}: {args.exc_value}
    {tb}
""")

    def stop(self):
        """Stop monitoring."""
        if not self._event.is_set():
            self._event.set()
            self.join()

        # Final sample
        sample2file(self._pid, fd=self._logfile)
        print(
            "\n".join(
                [
                    datetime.now().strftime(
                        "# MRIQC Resource recorder stopped (%Y/%m/%d; %H:%M:%S)"
                    )
                ]
            ),
            file=self._logfile,
        )
        self._logfile.flush()
        self._logfile.close()

        return self._logfile.name

    def run(self):
        """Core monitoring function, called by start()"""
        start_time = time()
        wait_til = start_time
        while not self._event.is_set():
            sample2file(self._pid, fd=self._logfile, timestamp=wait_til)
            wait_til += self._freq
            self._event.wait(max(0, wait_til - time()))
