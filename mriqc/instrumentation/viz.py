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
"""Visualizing resource recordings."""
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

_TIME_LABEL = "runtime"


def plot(filename, param="mem_vsm_mb", mask_processes=tuple(), out_file=None):
    """Plot a recording file."""
    data = pd.read_csv(filename, sep=r"\s+", comment="#")

    # Rebase all events to be relative to start and convert to seconds
    data["timestamp"] -= data["timestamp"][0]
    data["timestamp"] /= 1.0e9

    # Convert processes from rows to columns
    unified = pd.DataFrame({_TIME_LABEL: sorted(set(data["timestamp"].values))})

    pids = sorted(set(data["pid"].values))
    proc_names = []
    for pid in pids:
        pid_info = data[data["pid"] == pid]
        try:
            label = f"{pid_info['name'].values[0]}"
        except KeyError:
            label = f"{pid}"

        if label in mask_processes:
            continue

        rows = unified[_TIME_LABEL].isin(pid_info["timestamp"])
        if label not in unified.columns:
            proc_names.append(label)
            unified[label] = 0

        unified.loc[rows, label] += (
            pid_info[param].values
            / (1024 if param.startswith("mem") else 1)
        )

    unified[proc_names] = unified[proc_names].replace({0: np.nan})

    fig = plt.figure(figsize=(15, 10), facecolor="white")
    ax = plt.gca()
    _ = unified.plot.area(x=_TIME_LABEL, cmap="tab20", linewidth=0, ax=ax)
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    ax.set_xlabel("Run time (mm:ss)")
    if param.startswith("mem"):
        ax.set_ylabel("Memory fingerprint (GB)")
    elif param.startswith("cpu"):
        ax.set_ylabel("CPU utilization")
    else:
        ax.set_ylabel(param)

    ax.set_xticklabels(
        [f"{int(float(v) // 60)}:{int(float(v) % 60)}" for v in ax.get_xticks()]
    )
    if out_file is not None:
        fig.savefig(out_file, bbox_inches="tight", pad_inches=0, dpi=300)
        return

    return fig
