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
   :py:func:`~mriqc.workflows.functional.output.init_func_report_wf`.

This workflow is orchestrated by :py:func:`fmri_qc_workflow`.
"""

from nipype.interfaces import utility as niu
from nipype.pipeline import engine as pe
from niworkflows.utils.connections import pop_file as _pop

from mriqc import config
from mriqc.workflows.functional.output import init_func_report_wf
from mriqc.workflows.functional.base import hmc


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
    from nipype.interfaces.afni import TStat
    from niworkflows.interfaces.header import SanitizeImage

    from mriqc.interfaces.functional import SelectEcho
    from mriqc.messages import BUILDING_WORKFLOW

    modality = 'pet'

    mem_gb = config.workflow.biggest_file_gb[modality]
    dataset = config.workflow.inputs[modality]
    metadata = config.workflow.inputs_metadata[modality]
    entities = config.workflow.inputs_entities[modality]

    message = BUILDING_WORKFLOW.format(
        modality='pet',
        detail=f'for {len(dataset)} {modality.upper()} runs.',
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
        niu.IdentityInterface(fields=['qc', 'mosaic', 'out_group', 'out_dvars', 'out_fd']),
        name='outputnode',
    )

    sanitize = pe.MapNode(
        SanitizeImage(max_32bit=config.execution.float32),
        name='sanitize',
        mem_gb=mem_gb * 4.0,
        iterfield=['in_file'],
    )

    # Workflow --------------------------------------------------------

    # 1. HMC: head motion correct
    hmcwf = hmc(omp_nthreads=config.nipype.omp_nthreads)

    # Set HMC settings
    hmcwf.inputs.inputnode.fd_radius = config.workflow.fd_radius

    # 2. Compute mean fmri
    mean = pe.MapNode(
        TStat(options='-mean', outputtype='NIFTI_GZ'),
        name='mean',
        mem_gb=mem_gb * 1.5,
        iterfield=['in_file'],
    )

    # 7. Compute IQMs
    iqmswf = compute_iqms()
    # Reports
    func_report_wf = init_func_report_wf()

    # fmt: off

    workflow.connect([
        (inputnode, sanitize, [('in_file', 'in_file')]),
        (sanitize, hmcwf, [('out_file', 'inputnode.in_file')]),
        (hmcwf, mean, [('outputnode.out_file', 'in_file')]),
        # Feed IQMs computation
        (inputnode, iqmswf, [('in_file', 'inputnode.in_file'),
                             ('metadata', 'inputnode.metadata'),
                             ('entities', 'inputnode.entities')]),
        (sanitize, iqmswf, [('out_file', 'inputnode.in_ras')]),
        (mean, iqmswf, [('out_file', 'inputnode.epi_mean')]),
        (hmcwf, iqmswf, [('outputnode.out_file', 'inputnode.hmc_epi'),
                         ('outputnode.out_fd', 'inputnode.hmc_fd'),
                         ('outputnode.mpars', 'inputnode.mpars')]),
        # Feed reportlet generation
        (inputnode, func_report_wf, [
            ('in_file', 'inputnode.name_source'),
            ('metadata', 'inputnode.meta_sidecar'),
        ]),
        (sanitize, func_report_wf, [('out_file', 'inputnode.in_ras')]),
        (mean, func_report_wf, [('out_file', 'inputnode.epi_mean')]),
        (hmcwf, func_report_wf, [
            ('outputnode.out_fd', 'inputnode.hmc_fd'),
            ('outputnode.out_file', 'inputnode.hmc_epi'),
        ]),
        (iqmswf, func_report_wf, [
            ('outputnode.out_file', 'inputnode.in_iqms'),
            ('outputnode.out_dvars', 'inputnode.in_dvars'),
            ('outputnode.outliers', 'inputnode.outliers'),
        ]),
        (hmcwf, outputnode, [('outputnode.out_fd', 'out_fd')]),
    ])
    # fmt: on

    if config.workflow.fft_spikes_detector:
        # fmt: off
        workflow.connect([
            (iqmswf, func_report_wf, [
                ('outputnode.out_spikes', 'inputnode.in_spikes'),
                ('outputnode.out_fft', 'inputnode.in_fft'),
            ]),
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


def compute_iqms(name='ComputeIQMs'):
    """
    Initialize the workflow that actually computes the IQMs.

    .. workflow::

        from mriqc.workflows.functional.base import compute_iqms
        from mriqc.testing import mock_config
        with mock_config():
            wf = compute_iqms()

    """
    from nipype.algorithms.confounds import ComputeDVARS

    from mriqc.interfaces import DerivativesDataSink, FunctionalQC, GatherTimeseries, IQMFileSink
    from mriqc.interfaces.reports import AddProvenance
    from mriqc.workflows.utils import _tofloat, get_fwhmx

    mem_gb = config.workflow.biggest_file_gb['bold']

    workflow = pe.Workflow(name=name)
    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                'in_file',
                'metadata',
                'entities',
                'in_ras',
                'epi_mean',
                'brainmask',
                'hmc_epi',
                'hmc_fd',
                'fd_thres',
                'in_tsnr',
                'mpars',
                'exclude_index',
            ]
        ),
        name='inputnode',
    )
    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                'out_file',
                'out_dvars',
                'outliers',
                'out_spikes',
                'out_fft',
            ]
        ),
        name='outputnode',
    )

    # Set FD threshold
    inputnode.inputs.fd_thres = config.workflow.fd_thres

    # Compute DVARS
    dvnode = pe.MapNode(
        ComputeDVARS(save_plot=False, save_all=True),
        name='ComputeDVARS',
        mem_gb=mem_gb * 3,
        iterfield=['in_file'],
    )

    # AFNI quality measures
    fwhm = pe.MapNode(get_fwhmx(), name='smoothness', iterfield=['in_file'])
    fwhm.inputs.acf = True  # Only AFNI >= 16

    measures = pe.MapNode(
        FunctionalQC(),
        name='measures',
        mem_gb=mem_gb * 3,
        n_procs=max(1, config.nipype.nprocs // 2),
        iterfield=['in_epi', 'in_hmc', 'in_tsnr', 'in_dvars', 'in_fwhm'],
    )

    timeseries = pe.MapNode(
        GatherTimeseries(mpars_source='AFNI'),
        name='timeseries',
        mem_gb=mem_gb * 3,
        iterfield=['dvars', 'outliers', 'quality'],
    )

    # fmt: off
    workflow.connect([
        (inputnode, dvnode, [('hmc_epi', 'in_file'),
                             ('brainmask', 'in_mask')]),
        (inputnode, measures, [('epi_mean', 'in_epi'),
                               ('brainmask', 'in_mask'),
                               ('hmc_epi', 'in_hmc'),
                               ('hmc_fd', 'in_fd'),
                               ('fd_thres', 'fd_thres'),
                               ('in_tsnr', 'in_tsnr')]),
        (inputnode, fwhm, [('epi_mean', 'in_file'),
                           ('brainmask', 'mask')]),
        (dvnode, measures, [('out_all', 'in_dvars')]),
        (fwhm, measures, [(('fwhm', _tofloat), 'in_fwhm')]),
        (dvnode, outputnode, [('out_all', 'out_dvars')]),
        (dvnode, timeseries, [('out_all', 'dvars')]),
        (inputnode, timeseries, [('hmc_fd', 'fd'), ('mpars', 'mpars')]),
    ])
    # fmt: on

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

    # Save timeseries TSV file
    ds_timeseries = pe.MapNode(
        DerivativesDataSink(base_directory=str(config.execution.output_dir), suffix='timeseries'),
        name='ds_timeseries',
        run_without_submitting=True,
        iterfield=['in_file', 'source_file', 'meta_dict'],
    )

    # fmt: off
    workflow.connect([
        (inputnode, addprov, [('in_file', 'in_file')]),
        (inputnode, datasink, [('in_file', 'in_file'),
                               ('exclude_index', 'dummy_trs'),
                               ('entities', 'entities'),
                               ('metadata', 'metadata')]),
        (addprov, datasink, [('out_prov', 'provenance')]),
        (measures, datasink, [('out_qc', 'root')]),
        (datasink, outputnode, [('out_file', 'out_file')]),
        (inputnode, ds_timeseries, [('in_file', 'source_file')]),
        (timeseries, ds_timeseries, [('timeseries_file', 'in_file'),
                                     ('timeseries_metadata', 'meta_dict')]),
    ])
    # fmt: on

    return workflow


def _parse_tqual(in_file):
    if isinstance(in_file, (list, tuple)):
        return [_parse_tqual(f) for f in in_file] if len(in_file) > 1 else _parse_tqual(in_file[0])

    import numpy as np

    with open(in_file) as fin:
        lines = fin.readlines()
    return np.mean([float(line.strip()) for line in lines if not line.startswith('++')])


def _parse_tout(in_file):
    if isinstance(in_file, (list, tuple)):
        return [_parse_tout(f) for f in in_file] if len(in_file) > 1 else _parse_tout(in_file[0])

    import numpy as np

    data = np.loadtxt(in_file)  # pylint: disable=no-member
    return data.mean()