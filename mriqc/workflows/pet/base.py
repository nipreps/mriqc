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
"""
Functional workflow
===================

.. image :: _static/functional_workflow_source.svg

The functional workflow follows the following steps:

#. Sanitize (revise data types and xforms) input data, read
   associated metadata and discard non-steady state frames.
#. :abbr:`HMC (head-motion correction)` based on ``3dvolreg`` from
   AFNI -- :py:func:`hmc`.
#. Skull-stripping of the time-series (AFNI) --
   :py:func:`fmri_bmsk_workflow`.
#. Calculate mean time-series, and :abbr:`tSNR (temporal SNR)`.
#. Spatial Normalization to MNI (ANTs) -- :py:func:`epi_mni_align`
#. Extraction of IQMs -- :py:func:`compute_iqms`.
#. Individual-reports generation --
   :py:func:`~mriqc.workflows.functional.output.init_pet_report_wf`.

This workflow is orchestrated by :py:func:`fmri_qc_workflow`.
"""

from nipype.interfaces import utility as niu
from nipype.pipeline import engine as pe

from mriqc import config
from mriqc.workflows.pet.output import init_pet_report_wf


def pet_qc_workflow(name='petMRIQC'):
    """
    Initialize the (pet)MRIQC workflow.

    .. workflow::

        import os.path as op
        from mriqc.workflows.functional.base import pet_qc_workflow
        from mriqc.testing import mock_config
        with mock_config():
            wf = pet_qc_workflow()

    """
    from mriqc.messages import BUILDING_WORKFLOW

    dataset = config.workflow.inputs['pet']
    metadata = config.workflow.inputs_metadata['pet']
    entities = config.workflow.inputs_entities['pet']

    message = BUILDING_WORKFLOW.format(
        modality='pet',
        detail=f'for {len(dataset)} PET runs.',
    )
    config.loggers.workflow.info(message)

    # Define workflow, inputs and outputs
    # 0. Get data, put it in RAS orientation
    workflow = pe.Workflow(name=name)
    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=['in_file', 'metadata', 'entities'],
        ),
        name='inputnode',
    )
    inputnode.synchronize = True  # Do not test combinations of iterables
    inputnode.iterables = [
        ('in_file', dataset),
        ('metadata', metadata),
        ('entities', entities),
    ]

    outputnode = pe.Node(
        niu.IdentityInterface(fields=['qc', 'mosaic', 'out_group', 'out_fd']),
        name='outputnode',
    )

    # Workflow --------------------------------------------------------

    # 1. HMC: head motion correct
    hmcwf = hmc(omp_nthreads=config.nipype.omp_nthreads)

    # Set HMC settings
    hmcwf.inputs.inputnode.fd_radius = config.workflow.fd_radius

    # 7. Compute IQMs
    iqmswf = compute_iqms()
    # Reports
    pet_report_wf = init_pet_report_wf()

    # fmt: off
    workflow.connect([
        (inputnode, hmcwf, [('in_file', 'inputnode.in_file')]),
        # Feed IQMs computation
        (inputnode, iqmswf, [('in_file', 'inputnode.in_file'),
                             ('metadata', 'inputnode.metadata'),
                             ('entities', 'inputnode.entities')]),
        (hmcwf, iqmswf, [('outputnode.out_fd', 'inputnode.hmc_fd')]),
        # Feed reportlet generation
        (inputnode, pet_report_wf, [('in_file', 'inputnode.name_source')]),
        (hmcwf, pet_report_wf, [
            ('outputnode.out_fd', 'inputnode.hmc_fd'),
            ('outputnode.out_mot_param', 'inputnode.hmc_mot_param'),
        ]),
        (iqmswf, pet_report_wf, [
            ('outputnode.out_file', 'inputnode.in_iqms'),
        ]),
        (hmcwf, outputnode, [('outputnode.out_fd', 'out_fd')]),
    ])
    # fmt: on

    # Upload metrics
    if not config.execution.no_sub:
        from mriqc.interfaces.webapi import UploadIQMs

        upldwf = pe.MapNode(
            UploadIQMs(
                endpoint=config.execution.webapi_url,
                auth_token=config.execution.webapi_token,
                strict=config.execution.upload_strict,
            ),
            name='UploadMetrics',
            iterfield=['in_iqms'],
        )

        # fmt: off
        workflow.connect([
            (iqmswf, upldwf, [('outputnode.out_file', 'in_iqms')]),
        ])
        # fmt: on

    return workflow


def hmc(name='petHMC', omp_nthreads=None):
    """
    Create a :abbr:` petHMC (head motion correction)` workflow.

    .. workflow::

        from mriqc.workflows.functional.base import hmc
        from mriqc.testing import mock_config
        with mock_config():
            wf = hmc()

    """
    from nipype.algorithms.confounds import FramewiseDisplacement
    from nipype.interfaces.afni import Volreg
    from mriqc.interfaces.pet import ChooseRefHMC

    mem_gb = config.workflow.biggest_file_gb['pet']

    workflow = pe.Workflow(name=name)

    inputnode = pe.Node(
        niu.IdentityInterface(fields=['in_file', 'fd_radius']),
        name='inputnode',
    )

    outputnode = pe.Node(
        niu.IdentityInterface(fields=['out_file', 'out_mot_param', 'out_fd', 'mpars']),
        name='outputnode',
    )
    choose_ref_node = pe.Node(
        ChooseRefHMC(),
        name='ChooseRefHMC',
    )

    estimate_hm = pe.Node(
        Volreg(args='-Fourier -twopass', zpad=4, outputtype='NIFTI_GZ'),
        name='estimate_hm',
        mem_gb=mem_gb * 2.5,
    )

    # Compute the frame-wise displacement
    fdnode = pe.Node(
        FramewiseDisplacement(normalize=False, parameter_source='AFNI'),
        name='ComputeFD',
    )

    # fmt: off
    workflow.connect([
        (inputnode, choose_ref_node, [('in_file', 'in_file')]),
        (inputnode, estimate_hm, [('in_file', 'in_file')]),
        (inputnode, fdnode, [('fd_radius', 'radius')]),
        (choose_ref_node, estimate_hm, [('out_file', 'basefile')]),
        (estimate_hm, outputnode, [('oned_file', 'out_mot_param')]),
        (estimate_hm, fdnode, [('oned_file', 'in_file')]),
        (estimate_hm, outputnode, [('oned_file', 'mpars')]),
        (fdnode, outputnode, [('out_file', 'out_fd')]),
    ])
    # fmt: on

    return workflow


def compute_iqms(name='ComputeIQMs'):
    """
    Initialize the workflow that actually computes the IQMs.

    .. workflow::

        from mriqc.workflows.functional.base import compute_iqms
        from mriqc.testing import mock_config
        with mock_config():
            wf = compute_iqms()

    """
    from mriqc.interfaces import IQMFileSink
    from mriqc.interfaces.reports import AddProvenance
    from mriqc.interfaces.pet import FDStats

    workflow = pe.Workflow(name=name)
    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                'in_file',
                'metadata',
                'entities',
                'hmc_fd',
                'fd_thres',
            ]
        ),
        name='inputnode',
    )
    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                'out_file',
            ]
        ),
        name='outputnode',
    )

    # Set FD threshold
    inputnode.inputs.fd_thres = config.workflow.fd_thres

    # Compute FD statistics
    fd_stats = pe.Node(
        FDStats(),
        name='FDStats',
    )

    addprov = pe.MapNode(
        AddProvenance(modality='pet'),
        name='provenance',
        run_without_submitting=True,
        iterfield=['in_file'],
    )

    # Save to JSON file
    datasink = pe.MapNode(
        IQMFileSink(
            modality='pet',
            out_dir=str(config.execution.output_dir),
            dataset=config.execution.dsname,
        ),
        name='datasink',
        run_without_submitting=True,
        iterfield=['in_file', 'root', 'metadata', 'provenance'],
    )

    # fmt: off
    workflow.connect([
        (inputnode, addprov, [('in_file', 'in_file')]),
        (inputnode, datasink, [('in_file', 'in_file'),
                               ('entities', 'entities'),
                               ('metadata', 'metadata')]),
        (inputnode, fd_stats, [('hmc_fd', 'in_fd'),
                               ('fd_thres', 'fd_thres')]),
        (addprov, datasink, [('out_prov', 'provenance')]),
        (fd_stats, datasink, [('out_fd', 'root')]),
        (datasink, outputnode, [('out_file', 'out_file')]),
    ])
    # fmt: on

    return workflow
