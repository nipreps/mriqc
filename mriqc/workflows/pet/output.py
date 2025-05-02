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
"""Writing out functional reportlets."""
import os
from os import path as op
import numpy as np
import matplotlib.pyplot as plt

from nipype.interfaces import utility as niu
from nipype.pipeline import engine as pe

from mriqc import config
from mriqc.interfaces import DerivativesDataSink


def init_pet_report_wf(name='pet_report_wf'):
    """
    Write out individual reportlets.

    .. workflow::

        from mriqc.workflows.functional.output import init_pet_report_wf
        from mriqc.testing import mock_config
        with mock_config():
            wf = init_pet_report_wf()

    """
    from nipype.interfaces.fsl import AvScale

    reportlets_dir = config.execution.work_dir / 'reportlets'

    workflow = pe.Workflow(name=name)
    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                'hmc_epi',
                'hmc_xfm',
                'hmc_fd',
                'in_iqms',
                'name_source',
            ]
        ),
        name='inputnode',
    )

    est_trans_rot = pe.MapNode(
        interface=AvScale(all_param=True),
        name="est_trans_rot",
        iterfield=["mat_file"],
    )
    compute_hmc_stat = pe.Node(
        niu.Function(
            input_names=["translations", "rotation_translation_matrix", "in_file"],
            output_names=["max_x", "max_y", "max_z", "max_tot", "median_tot"],
            function=compute_hmc_stat,
        ),
        name="compute_hmc_stat",
    )

    plot_fd = pe.Node(
        niu.Function(
            input_names=["fd", "in_file"],
            output_names=["out_file"],
            function=plot_fd,
        ),
        name="plot_fd",
    )

    plot_trans = pe.Node(
        niu.Function(
            input_names=["translations", "in_file"],
            output_names=["out_file"],
            function=plot_translation,
        ),
        name="plot_translation",
    )

    plot_rot = pe.Node(
        niu.Function(
            input_names=["rot_angles", "in_file"],
            output_names=["out_file"],
            function=plot_rotation,
        ),
        name="plot_rotation",
    )

    ds_report_fd = pe.MapNode(
        DerivativesDataSink(
            base_directory=reportlets_dir,
            desc='fd',
            datatype='figures',
            dismiss_entities=('part',),
        ),
        name='ds_report_fd',
        run_without_submitting=True,
        iterfield=['in_file', 'source_file'],
    )

    ds_report_trans = pe.MapNode(
        DerivativesDataSink(
            base_directory=reportlets_dir,
            desc='translation',
            datatype='figures',
            dismiss_entities=('part',),
        ),
        name='ds_report_trans',
        run_without_submitting=True,
        iterfield=['in_file', 'source_file'],
    )

    ds_report_rot = pe.MapNode(
        DerivativesDataSink(
            base_directory=reportlets_dir,
            desc='rotation',
            datatype='figures',
            dismiss_entities=('part',),
        ),
        name='ds_report_rot',
        run_without_submitting=True,
        iterfield=['in_file', 'source_file'],
    )

    # fmt: off
    workflow.connect([
        # (inputnode, rnode, [("in_iqms", "in_iqms")]),
        (inputnode, est_trans_rot, [("hmc_xfm", "mat_file")]),
        (inputnode, est_trans_rot, [("hmc_fd", "fd")]),
        (inputnode, ds_report_fd, [('name_source', 'source_file')]),
        (inputnode, ds_report_trans, [('name_source', 'source_file')]),
        (inputnode, ds_report_rot, [('name_source', 'source_file')]),
        (est_trans_rot, plot_rot, [('rot_angles', 'rot_angles')]),
        (est_trans_rot, plot_trans, [('translations', 'translations')]),
        (plot_fd, ds_report_fd, [('out_file', 'in_file')]),
        (plot_trans, ds_report_trans, [('out_file', 'in_file')]),
        (plot_rot, ds_report_rot, [('out_file', 'in_file')]),    
    ])
    # fmt: on

    return workflow

def plot_fd(fd, in_file):
    if out_file is None:
        fname, ext = op.splitext(op.basename(in_file))
        if ext == ".gz":
            fname, _ = op.splitext(fname)
        out_file = op.abspath(fname + "_fd.png")

    plt.figure(figsize=(11, 5))
    plt.plot(np.arange(0, len(fd)), fd, "-r")
    plt.legend(loc="upper left")
    plt.ylabel("Framewise Displacement [mm]")
    plt.xlabel("frame #")
    plt.grid(visible=True)
    plt.savefig(out_file, format="png")
    plt.close()

    return out_file

def plot_rotation(rot_angles, in_file):
    rot_angles_np = np.array(rot_angles)
    n_frames = rot_angles_np.shape[0]

    if out_file is None:
        fname, ext = op.splitext(op.basename(in_file))
        if ext == ".gz":
            fname, _ = op.splitext(fname)
        out_file = op.abspath(fname + "_rotation.png")

    plt.figure(figsize=(11, 5))
    plt.plot(np.arange(0, n_frames), rot_angles_np[:, 0], "-r", label="rot_x")
    plt.plot(np.arange(0, n_frames), rot_angles_np[:, 1], "-g", label="rot_y")
    plt.plot(np.arange(0, n_frames), rot_angles_np[:, 2], "-b", label="rot_z")
    plt.legend(loc="upper left")
    plt.ylabel("Rotation [degrees]")
    plt.xlabel("frame #")
    plt.grid(visible=True)
    plt.savefig(out_file, format="png")
    plt.close()

    return out_file

    
def plot_translation(translations, in_file):
    """
    Function to plot estimated motion data

    :in_file : list of estimated motion data
    :type in_file : list

    :return : Plots of estimated motion data
    :rtype : png
    """
    translations_np = np.array(translations)
    n_frames = translations_np.shape[0]

    if out_file is None:
        fname, ext = op.splitext(op.basename(in_file))
        if ext == ".gz":
            fname, _ = op.splitext(fname)
        out_file = op.abspath(fname + "_translation.png")

    plt.figure(figsize=(11, 5))
    plt.plot(np.arange(0, n_frames), translations_np[:, 0], "-r", label="trans_x")
    plt.plot(np.arange(0, n_frames), translations_np[:, 1], "-g", label="trans_y")
    plt.plot(np.arange(0, n_frames), translations_np[:, 2], "-b", label="trans_z")
    plt.legend(loc="upper left")
    plt.ylabel("Translation [mm]")
    plt.xlabel("frame #")
    plt.grid(visible=True)
    plt.savefig(out_file, format="png")
    plt.close()

    return out_file

    
