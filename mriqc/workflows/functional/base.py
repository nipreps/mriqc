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


def fmri_qc_workflow(name='funcMRIQC'):
    """
    Initialize the (f)MRIQC workflow.

    .. workflow::

        import os.path as op
        from mriqc.workflows.functional.base import fmri_qc_workflow
        from mriqc.testing import mock_config
        with mock_config():
            wf = fmri_qc_workflow()

    """
    from nipype.algorithms.confounds import TSNR, NonSteadyStateDetector
    from nipype.interfaces.afni import TStat
    from niworkflows.interfaces.header import SanitizeImage

    from mriqc.interfaces.functional import SelectEcho
    from mriqc.messages import BUILDING_WORKFLOW

    mem_gb = config.workflow.biggest_file_gb['bold']
    dataset = config.workflow.inputs['bold']
    metadata = config.workflow.inputs_metadata['bold']
    entities = config.workflow.inputs_entities['bold']

    message = BUILDING_WORKFLOW.format(
        modality='functional',
        detail=f'for {len(dataset)} BOLD runs.',
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

    pick_echo = pe.Node(SelectEcho(), name='pick_echo')

    non_steady_state_detector = pe.Node(NonSteadyStateDetector(), name='non_steady_state_detector')

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

    # Compute TSNR using nipype implementation
    tsnr = pe.MapNode(
        TSNR(),
        name='compute_tsnr',
        mem_gb=mem_gb * 2.5,
        iterfield=['in_file'],
    )

    # EPI to MNI registration
    ema = epi_mni_align()

    # 7. Compute IQMs
    iqmswf = compute_iqms()
    # Reports
    func_report_wf = init_func_report_wf()

    # fmt: off

    workflow.connect([
        (inputnode, pick_echo, [('in_file', 'in_files'),
                                ('metadata', 'metadata')]),
        (inputnode, sanitize, [('in_file', 'in_file')]),
        (pick_echo, non_steady_state_detector, [('out_file', 'in_file')]),
        (non_steady_state_detector, sanitize, [('n_volumes_to_discard', 'n_volumes_to_discard')]),
        (sanitize, hmcwf, [('out_file', 'inputnode.in_file')]),
        (hmcwf, mean, [('outputnode.out_file', 'in_file')]),
        (hmcwf, tsnr, [('outputnode.out_file', 'in_file')]),
        (mean, ema, [(('out_file', _pop), 'inputnode.epi_mean')]),
        # Feed IQMs computation
        (inputnode, iqmswf, [('in_file', 'inputnode.in_file'),
                             ('metadata', 'inputnode.metadata'),
                             ('entities', 'inputnode.entities')]),
        (sanitize, iqmswf, [('out_file', 'inputnode.in_ras')]),
        (mean, iqmswf, [('out_file', 'inputnode.epi_mean')]),
        (hmcwf, iqmswf, [('outputnode.out_file', 'inputnode.hmc_epi'),
                         ('outputnode.out_fd', 'inputnode.hmc_fd'),
                         ('outputnode.mpars', 'inputnode.mpars')]),
        (tsnr, iqmswf, [('tsnr_file', 'inputnode.in_tsnr')]),
        (non_steady_state_detector, iqmswf, [('n_volumes_to_discard', 'inputnode.exclude_index')]),
        # Feed reportlet generation
        (inputnode, func_report_wf, [
            ('in_file', 'inputnode.name_source'),
            ('metadata', 'inputnode.meta_sidecar'),
        ]),
        (sanitize, func_report_wf, [('out_file', 'inputnode.in_ras')]),
        (mean, func_report_wf, [('out_file', 'inputnode.epi_mean')]),
        (tsnr, func_report_wf, [('stddev_file', 'inputnode.in_stddev')]),
        (hmcwf, func_report_wf, [
            ('outputnode.out_fd', 'inputnode.hmc_fd'),
            ('outputnode.out_file', 'inputnode.hmc_epi'),
        ]),
        (ema, func_report_wf, [
            ('outputnode.epi_parc', 'inputnode.epi_parc'),
            ('outputnode.report', 'inputnode.mni_report'),
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

    # population specific changes to brain masking
    if config.workflow.species == 'human':
        from mriqc.workflows.shared import synthstrip_wf as fmri_bmsk_workflow

        skullstrip_epi = fmri_bmsk_workflow(omp_nthreads=config.nipype.omp_nthreads)
        # fmt: off
        workflow.connect([
            (mean, skullstrip_epi, [(('out_file', _pop), 'inputnode.in_files')]),
            (skullstrip_epi, ema, [('outputnode.out_mask', 'inputnode.epi_mask')]),
            (skullstrip_epi, iqmswf, [('outputnode.out_mask', 'inputnode.brainmask')]),
            (skullstrip_epi, func_report_wf, [('outputnode.out_mask', 'inputnode.brainmask')]),
        ])
        # fmt: on
    else:
        from mriqc.workflows.anatomical.base import _binarize

        binarise_labels = pe.Node(
            niu.Function(
                input_names=['in_file', 'threshold'],
                output_names=['out_file'],
                function=_binarize,
            ),
            name='binarise_labels',
        )

        # fmt: off
        workflow.connect([
            (ema, binarise_labels, [('outputnode.epi_parc', 'in_file')]),
            (binarise_labels, iqmswf, [('out_file', 'inputnode.brainmask')]),
            (binarise_labels, func_report_wf, [('out_file', 'inputnode.brainmask')])
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
    from nipype.interfaces.afni import OutlierCount, QualityIndex

    from mriqc.interfaces import DerivativesDataSink, FunctionalQC, GatherTimeseries, IQMFileSink
    from mriqc.interfaces.reports import AddProvenance
    from mriqc.interfaces.transitional import GCOR
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

    outliers = pe.MapNode(
        OutlierCount(fraction=True, out_file='outliers.out'),
        name='outliers',
        mem_gb=mem_gb * 2.5,
        iterfield=['in_file'],
    )

    quality = pe.MapNode(
        QualityIndex(automask=True),
        out_file='quality.out',
        name='quality',
        mem_gb=mem_gb * 3,
        iterfield=['in_file'],
    )

    gcor = pe.MapNode(GCOR(), name='gcor', mem_gb=mem_gb * 2, iterfield=['in_file'])

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
        (inputnode, quality, [('hmc_epi', 'in_file')]),
        (inputnode, outliers, [('hmc_epi', 'in_file'),
                               ('brainmask', 'mask')]),
        (inputnode, gcor, [('hmc_epi', 'in_file'),
                           ('brainmask', 'mask')]),
        (dvnode, measures, [('out_all', 'in_dvars')]),
        (fwhm, measures, [(('fwhm', _tofloat), 'in_fwhm')]),
        (dvnode, outputnode, [('out_all', 'out_dvars')]),
        (outliers, outputnode, [('out_file', 'outliers')]),
        (outliers, timeseries, [('out_file', 'outliers')]),
        (quality, timeseries, [('out_file', 'quality')]),
        (dvnode, timeseries, [('out_all', 'dvars')]),
        (inputnode, timeseries, [('hmc_fd', 'fd'), ('mpars', 'mpars')]),
    ])
    # fmt: on

    addprov = pe.MapNode(
        AddProvenance(modality='bold'),
        name='provenance',
        run_without_submitting=True,
        iterfield=['in_file'],
    )

    # Save to JSON file
    datasink = pe.MapNode(
        IQMFileSink(
            modality='bold',
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
        (outliers, datasink, [(('out_file', _parse_tout), 'aor')]),
        (gcor, datasink, [(('out', _tofloat), 'gcor')]),
        (quality, datasink, [(('out_file', _parse_tqual), 'aqi')]),
        (measures, datasink, [('out_qc', 'root')]),
        (datasink, outputnode, [('out_file', 'out_file')]),
        (inputnode, ds_timeseries, [('in_file', 'source_file')]),
        (timeseries, ds_timeseries, [('timeseries_file', 'in_file'),
                                     ('timeseries_metadata', 'meta_dict')]),
    ])
    # fmt: on

    # FFT spikes finder
    if config.workflow.fft_spikes_detector:
        from mriqc.workflows.utils import slice_wise_fft

        spikes_fft = pe.MapNode(
            niu.Function(
                input_names=['in_file'],
                output_names=['n_spikes', 'out_spikes', 'out_fft'],
                function=slice_wise_fft,
            ),
            name='SpikesFinderFFT',
            iterfield=['in_file'],
        )

        # fmt: off
        workflow.connect([
            (inputnode, spikes_fft, [('in_ras', 'in_file')]),
            (spikes_fft, outputnode, [('out_spikes', 'out_spikes'),
                                      ('out_fft', 'out_fft')]),
            (spikes_fft, datasink, [('n_spikes', 'spikes_num')])
        ])
        # fmt: on

    return workflow


def fmri_bmsk_workflow(name='fMRIBrainMask'):
    """
    Compute a brain mask for the input :abbr:`fMRI (functional MRI)` dataset.

    .. workflow::

        from mriqc.workflows.functional.base import fmri_bmsk_workflow
        from mriqc.testing import mock_config
        with mock_config():
            wf = fmri_bmsk_workflow()


    """
    from nipype.interfaces.afni import Automask

    workflow = pe.Workflow(name=name)
    inputnode = pe.Node(niu.IdentityInterface(fields=['in_file']), name='inputnode')
    outputnode = pe.Node(niu.IdentityInterface(fields=['out_file']), name='outputnode')
    afni_msk = pe.Node(Automask(outputtype='NIFTI_GZ'), name='afni_msk')

    # Connect brain mask extraction
    # fmt: off
    workflow.connect([
        (inputnode, afni_msk, [('in_file', 'in_file')]),
        (afni_msk, outputnode, [('out_file', 'out_file')])
    ])
    # fmt: on
    return workflow


def hmc(name='fMRI_HMC', omp_nthreads=None):
    """
    Create a :abbr:`HMC (head motion correction)` workflow for fMRI.

    .. workflow::

        from mriqc.workflows.functional.base import hmc
        from mriqc.testing import mock_config
        with mock_config():
            wf = hmc()

    """
    from nipype.algorithms.confounds import FramewiseDisplacement
    from nipype.interfaces.afni import Despike, Refit, Volreg

    mem_gb = config.workflow.biggest_file_gb['bold']

    workflow = pe.Workflow(name=name)

    inputnode = pe.Node(
        niu.IdentityInterface(fields=['in_file', 'fd_radius']),
        name='inputnode',
    )

    outputnode = pe.Node(
        niu.IdentityInterface(fields=['out_file', 'out_fd', 'mpars']),
        name='outputnode',
    )

    # calculate hmc parameters
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

    # Apply transforms to other echos
    apply_hmc = pe.MapNode(
        niu.Function(
            function=_apply_transforms,
            input_names=['in_file', 'in_xfm', 'max_concurrent'],
        ),
        name='apply_hmc',
        iterfield=['in_file'],
        # NiTransforms is a memory hog, so ensure only one process is running at a time
        n_procs=config.environment.cpu_count,
    )
    apply_hmc.inputs.max_concurrent = 4

    # fmt: off
    workflow.connect([
        (inputnode, fdnode, [('fd_radius', 'radius')]),
        (estimate_hm, apply_hmc, [('oned_matrix_save', 'in_xfm')]),
        (apply_hmc, outputnode, [('out', 'out_file')]),
        (estimate_hm, fdnode, [('oned_file', 'in_file')]),
        (estimate_hm, outputnode, [('oned_file', 'mpars')]),
        (fdnode, outputnode, [('out_file', 'out_fd')]),
    ])
    # fmt: on

    if not (config.workflow.despike or config.workflow.deoblique):
        # fmt: off
        workflow.connect([
            (inputnode, estimate_hm, [(('in_file', _pop), 'in_file')]),
            (inputnode, apply_hmc, [('in_file', 'in_file')]),
        ])
        # fmt: on
        return workflow

    # despiking, and deoblique
    deoblique_node = pe.MapNode(
        Refit(deoblique=True),
        name='deoblique',
        iterfield=['in_file'],
    )
    despike_node = pe.MapNode(
        Despike(outputtype='NIFTI_GZ'),
        name='despike',
        iterfield=['in_file'],
    )
    if config.workflow.despike and config.workflow.deoblique:
        # fmt: off
        workflow.connect([
            (inputnode, despike_node, [('in_file', 'in_file')]),
            (despike_node, deoblique_node, [('out_file', 'in_file')]),
            (deoblique_node, estimate_hm, [(('out_file', _pop), 'in_file')]),
            (deoblique_node, apply_hmc, [('out_file', 'in_file')]),
        ])
        # fmt: on
    elif config.workflow.despike:
        # fmt: off
        workflow.connect([
            (inputnode, despike_node, [('in_file', 'in_file')]),
            (despike_node, estimate_hm, [(('out_file', _pop), 'in_file')]),
            (despike_node, apply_hmc, [('out_file', 'in_file')]),
        ])
        # fmt: on
    elif config.workflow.deoblique:
        # fmt: off
        workflow.connect([
            (inputnode, deoblique_node, [('in_file', 'in_file')]),
            (deoblique_node, estimate_hm, [(('out_file', _pop), 'in_file')]),
            (deoblique_node, apply_hmc, [('out_file', 'in_file')]),
        ])
        # fmt: on
    else:
        raise NotImplementedError

    return workflow


def epi_mni_align(name='SpatialNormalization'):
    """
    Estimate the transform that maps the EPI space into MNI152NLin2009cAsym.

    The input epi_mean is the averaged and brain-masked EPI timeseries

    Returns the EPI mean resampled in MNI space (for checking out registration) and
    the associated "lobe" parcellation in EPI space.

    .. workflow::

        from mriqc.workflows.functional.base import epi_mni_align
        from mriqc.testing import mock_config
        with mock_config():
            wf = epi_mni_align()

    """
    from nipype.interfaces.ants import ApplyTransforms, N4BiasFieldCorrection
    from niworkflows.interfaces.reportlets.registration import (
        SpatialNormalizationRPT as RobustMNINormalization,
    )
    from templateflow.api import get as get_template

    # Get settings
    testing = config.execution.debug
    n_procs = config.nipype.nprocs
    ants_nthreads = config.nipype.omp_nthreads

    workflow = pe.Workflow(name=name)
    inputnode = pe.Node(
        niu.IdentityInterface(fields=['epi_mean', 'epi_mask']),
        name='inputnode',
    )
    outputnode = pe.Node(
        niu.IdentityInterface(fields=['epi_mni', 'epi_parc', 'report']),
        name='outputnode',
    )

    n4itk = pe.Node(N4BiasFieldCorrection(dimension=3, copy_header=True), name='SharpenEPI')

    norm = pe.Node(
        RobustMNINormalization(
            explicit_masking=False,
            flavor='testing' if testing else 'precise',
            float=config.execution.ants_float,
            generate_report=True,
            moving='boldref',
            num_threads=ants_nthreads,
            reference='boldref',
            template=config.workflow.template_id,
        ),
        name='EPI2MNI',
        num_threads=n_procs,
        mem_gb=3,
    )

    if config.workflow.species.lower() == 'human':
        norm.inputs.reference_image = str(
            get_template(config.workflow.template_id, resolution=2, suffix='boldref')
        )
        norm.inputs.reference_mask = str(
            get_template(
                config.workflow.template_id,
                resolution=2,
                desc='brain',
                suffix='mask',
            )
        )
    # adapt some population-specific settings
    else:
        from nirodents.workflows.brainextraction import _bspline_grid

        n4itk.inputs.shrink_factor = 1
        n4itk.inputs.n_iterations = [50] * 4
        norm.inputs.reference_image = str(get_template(config.workflow.template_id, suffix='T2w'))
        norm.inputs.reference_mask = str(
            get_template(
                config.workflow.template_id,
                desc='brain',
                suffix='mask',
            )[0]
        )

        bspline_grid = pe.Node(niu.Function(function=_bspline_grid), name='bspline_grid')

        # fmt: off
        workflow.connect([
            (inputnode, bspline_grid, [('epi_mean', 'in_file')]),
            (bspline_grid, n4itk, [('out', 'args')])
        ])
        # fmt: on

    # Warp segmentation into EPI space
    invt = pe.Node(
        ApplyTransforms(
            float=True,
            dimension=3,
            default_value=0,
            interpolation='MultiLabel',
        ),
        name='ResampleSegmentation',
    )

    if config.workflow.species.lower() == 'human':
        invt.inputs.input_image = str(
            get_template(
                config.workflow.template_id,
                resolution=1,
                desc='carpet',
                suffix='dseg',
            )
        )
    else:
        invt.inputs.input_image = str(
            get_template(
                config.workflow.template_id,
                suffix='dseg',
            )[-1]
        )

    # fmt: off
    workflow.connect([
        (inputnode, invt, [('epi_mean', 'reference_image')]),
        (inputnode, n4itk, [('epi_mean', 'input_image')]),
        (n4itk, norm, [('output_image', 'moving_image')]),
        (norm, invt, [
            ('inverse_composite_transform', 'transforms')]),
        (invt, outputnode, [('output_image', 'epi_parc')]),
        (norm, outputnode, [('warped_image', 'epi_mni'),
                            ('out_report', 'report')]),
    ])
    # fmt: on

    if config.workflow.species.lower() == 'human':
        workflow.connect([(inputnode, norm, [('epi_mask', 'moving_mask')])])

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


def _apply_transforms(in_file, in_xfm, max_concurrent):
    from pathlib import Path

    from nitransforms.linear import load
    from nitransforms.resampling import apply

    from mriqc.utils.bids import derive_bids_fname

    realigned = apply(
        load(in_xfm, fmt='afni', reference=in_file, moving=in_file),
        in_file,
        dtype_width=4,
        serialize_nvols=2,
        max_concurrent=max_concurrent,
        mode='reflect',
    )
    out_file = derive_bids_fname(
        in_file,
        entity='desc-realigned',
        newpath=Path.cwd(),
        absolute=True,
    )

    realigned.to_filename(out_file)
    return str(out_file)
