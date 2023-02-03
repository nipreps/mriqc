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
from time import time_ns, sleep
from datetime import datetime
from pathlib import Path
from multiprocessing import Process, Event
from contextlib import suppress
import signal
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
    exclude=tuple(),
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
        if process.pid in exclude:
            continue
        with suppress(psutil.NoSuchProcess):
            proc_info.append(process.as_dict(attrs=attrs))

    return proc_info


def parse_sample(datapoint, timestamp=None, attrs=SAMPLE_ATTRS):
    """Convert a sample dictionary into a list of string values."""
    retval = [f"{timestamp or time_ns()}"]

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


def sample2file(
    pid=None, recursive=True, timestamp=None, fd=None, flush=True, exclude=tuple()
):
    if fd is None:
        return

    print(
        "\n".join(
            [
                "\t".join(parse_sample(s, timestamp=timestamp))
                for s in sample(pid=pid, recursive=recursive, exclude=exclude)
            ]
        ),
        file=fd,
    )
    if flush:
        fd.flush()


class ResourceRecorder(Process):
    """Attach a ``Thread`` to sample a specific PID with a certain frequence."""

    def __init__(
        self, pid, frequency=0.2, log_file=None, exclude_probe=True, **process_kwargs
    ):
        Process.__init__(self, name="nipype_resmon", daemon=True, **process_kwargs)

        self._pid = pid
        """The process to be sampled."""
        self._logfile = str(
            Path(log_file if log_file is not None else f".prof-{pid}.tsv").absolute()
        )
        """An open file descriptor where results are dumped."""
        self._exclude = exclude_probe or tuple()
        """A list/tuple containing PIDs that should not be monitored."""
        self._freq_ns = int(max(frequency, 0.02) * 1e9)
        """Sampling frequency (stored in ns)."""
        self._done = Event()
        """Flag indicating if the process is marked to finish."""

        signal.signal(signal.SIGINT, self.stop)
        signal.signal(signal.SIGTERM, self.stop)

    def run(self, *args, **kwargs):
        """Core monitoring function, called by start()"""

        # Open file now, because it cannot be pickled.
        Path(self._logfile).parent.mkdir(parents=True, exist_ok=True)
        _logfile = Path(self._logfile).open("w")

        # Write headers (comment trace + header row)
        _header = [
            f"# MRIQC Resource recorder started tracking PID {self._pid} "
            f"{datetime.now().strftime('(%Y/%m/%d; %H:%M:%S)')}",
            "\t".join(("timestamp", *SAMPLE_ATTRS)).replace(
                "memory_info", "mem_rss_mb\tmem_vsm_mb"
            ),
        ]
        print("\n".join(_header), file=_logfile)

        # Add self to exclude list if pertinent
        if self._exclude is True:
            self._exclude = (psutil.Process().pid,)

        # Ensure done is not marked set
        self._done.clear()

        # Initiate periodic sampling
        start_time = time_ns()
        wait_til = start_time
        while not self._done.is_set():
            try:
                sample2file(self._pid, fd=_logfile, timestamp=wait_til)
            except psutil.NoSuchProcess:
                print(
                    f"# MRIQC Resource recorder killed "
                    f"{datetime.now().strftime('(%Y/%m/%d; %H:%M:%S)')}",
                    file=_logfile,
                )
                _logfile.flush()
                _logfile.close()
                break

            wait_til += self._freq_ns
            sleep(max(0, (wait_til - time_ns()) / 1.0e9))

        _logfile.close()

    def stop(self, *args):
        # Tear-down process
        self._done.set()
        with Path(self._logfile).open("a") as f:
            f.write(
                f"# MRIQC Resource recorder finished "
                f"{datetime.now().strftime('(%Y/%m/%d; %H:%M:%S)')}",
            )
