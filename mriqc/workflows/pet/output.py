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

from nipype.interfaces import utility as niu
from nipype.pipeline import engine as pe

from mriqc import config
from mriqc.interfaces import DerivativesDataSink
from mriqc.qc.pet import PlotFD, PlotRotation, PlotTranslation


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
                'hmc_mot_param',
                'hmc_fd',
                'in_iqms',
                'name_source',
            ]
        ),
        name='inputnode',
    )

    plot_fd = pe.Node(
        PlotFD(),
        name='plot_fd',
    )

    plot_trans = pe.Node(
        PlotTranslation(),
        name='plot_translation',
    )

    plot_rot = pe.Node(
        PlotRotation(),
        name='plot_rotation',
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
        (inputnode, plot_fd, [('hmc_fd', 'in_fd')]),
        (inputnode, plot_fd, [('name_source', 'in_file')]),
        (inputnode, plot_trans, [('name_source', 'in_file'),
                                 ('hmc_mot_param', 'mot_param')]),
        (inputnode, plot_rot, [('name_source', 'in_file'),
                               ('hmc_mot_param', 'mot_param')]),
        (inputnode, ds_report_fd, [('name_source', 'source_file')]),
        (inputnode, ds_report_trans, [('name_source', 'source_file')]),
        (inputnode, ds_report_rot, [('name_source', 'source_file')]),
        (plot_fd, ds_report_fd, [('out_file', 'in_file')]),
        (plot_trans, ds_report_trans, [('out_file', 'in_file')]),
        (plot_rot, ds_report_rot, [('out_file', 'in_file')]),    
    ])
    # fmt: on

    return workflow
