# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# Copyright 2023 The NiPreps Developers <nipreps@gmail.com>
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
Diffusion MRI workflow
======================

.. image :: _static/diffusion_workflow_source.svg

The diffusion workflow follows the following steps:

#. Sanitize (revise data types and xforms) input data, read
   associated metadata and discard non-steady state frames.
#. :abbr:`HMC (head-motion correction)` based on ``3dvolreg`` from
   AFNI -- :py:func:`hmc`.
#. Skull-stripping of the time-series (AFNI) --
   :py:func:`dmri_bmsk_workflow`.
#. Calculate mean time-series, and :abbr:`tSNR (temporal SNR)`.
#. Spatial Normalization to MNI (ANTs) -- :py:func:`epi_mni_align`
#. Extraction of IQMs -- :py:func:`compute_iqms`.
#. Individual-reports generation --
   :py:func:`~mriqc.workflows.diffusion.output.init_dwi_report_wf`.

This workflow is orchestrated by :py:func:`dmri_qc_workflow`.
"""

from nipype.interfaces import utility as niu
from nipype.pipeline import engine as pe

from mriqc import config
from mriqc.workflows.diffusion.output import init_dwi_report_wf

DEFAULT_MEMORY_MIN_GB = 0.01


def dmri_qc_workflow(name='dwiMRIQC'):
    """
    Initialize the dMRI-QC workflow.

    .. workflow::

        import os.path as op
        from mriqc.workflows.diffusion.base import dmri_qc_workflow
        from mriqc.testing import mock_config
        with mock_config():
            wf = dmri_qc_workflow()

    """
    from nipype.interfaces.afni import Volreg
    from nipype.interfaces.mrtrix3.preprocess import DWIDenoise
    from niworkflows.interfaces.header import SanitizeImage
    from niworkflows.interfaces.images import RobustAverage

    from mriqc.interfaces.diffusion import (
        PIESNO,
        CCSegmentation,
        CorrectSignalDrift,
        DiffusionModel,
        ExtractOrientations,
        NumberOfShells,
        ReadDWIMetadata,
        SpikingVoxelsMask,
        WeightedStat,
    )
    from mriqc.messages import BUILDING_WORKFLOW
    from mriqc.workflows.shared import synthstrip_wf as dmri_bmsk_workflow

    # Enable if necessary
    # mem_gb = config.workflow.biggest_file_gb['dwi']
    dataset = config.workflow.inputs['dwi']
    metadata = config.workflow.inputs_metadata['dwi']
    entities = config.workflow.inputs_entities['dwi']
    message = BUILDING_WORKFLOW.format(
        modality='diffusion',
        detail=f'for {len(dataset)} NIfTI files.',
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

    sanitize = pe.Node(
        SanitizeImage(
            n_volumes_to_discard=0,
            max_32bit=config.execution.float32,
        ),
        name='sanitize',
        mem_gb=4.0,
    )

    # Workflow --------------------------------------------------------

    # Read metadata & bvec/bval, estimate number of shells, extract and split B0s
    load_bmat = pe.Node(
        ReadDWIMetadata(index_db=config.execution.bids_database_dir),
        name='load_bmat',
    )
    shells = pe.Node(NumberOfShells(), name='shells')
    get_lowb = pe.Node(
        ExtractOrientations(),
        name='get_lowb',
        n_procs=max(1, config.nipype.nprocs // 2),
    )

    # Generate B0 reference
    dwi_ref = pe.Node(
        RobustAverage(mc_method=None),
        name='dwi_ref',
        n_procs=max(1, config.nipype.nprocs // 2),
    )

    hmc_b0 = pe.Node(
        Volreg(args='-Fourier -twopass', zpad=4, outputtype='NIFTI_GZ'),
        name='hmc_b0',
        mem_gb=3.0,
        n_procs=config.nipype.nprocs,
    )

    # Calculate brainmask
    dmri_bmsk = dmri_bmsk_workflow(omp_nthreads=config.nipype.omp_nthreads)

    # HMC: head motion correct
    hmcwf = hmc_workflow()

    get_hmc_shells = pe.MapNode(
        ExtractOrientations(),
        name='get_hmc_shells',
        iterfield=['indices'],
    )

    # Split shells and compute some stats
    averages = pe.MapNode(
        WeightedStat(),
        name='averages',
        n_procs=max(1, config.nipype.nprocs // 2),
        iterfield=['in_weights'],
    )
    stddev = pe.MapNode(
        WeightedStat(stat='std'),
        name='stddev',
        n_procs=max(1, config.nipype.nprocs // 2),
        iterfield=['in_weights'],
    )

    dwidenoise = pe.Node(
        DWIDenoise(
            noise='noisemap.nii.gz',
            nthreads=config.nipype.omp_nthreads,
        ),
        name='dwidenoise',
        n_procs=config.nipype.nprocs,
    )
    drift = pe.Node(
        CorrectSignalDrift(),
        name='drift',
        n_procs=max(1, config.nipype.nprocs // 2),
    )
    sp_mask = pe.Node(
        SpikingVoxelsMask(),
        name='sp_mask',
        n_procs=max(1, config.nipype.nprocs // 2),
    )

    # Fit DTI/DKI model
    dwimodel = pe.Node(
        DiffusionModel(),
        name='dwimodel',
        n_procs=config.nipype.nprocs,
    )

    # Calculate CC mask
    cc_mask = pe.Node(
        CCSegmentation(),
        name='cc_mask',
        n_procs=max(1, config.nipype.nprocs // 2),
    )

    # Run PIESNO noise estimation
    piesno = pe.Node(
        PIESNO(),
        name='piesno',
        n_procs=max(1, config.nipype.nprocs // 2),
    )

    # EPI to MNI registration
    spatial_norm = epi_mni_align()

    # Compute IQMs
    iqms_wf = compute_iqms()

    # Generate outputs
    dwi_report_wf = init_dwi_report_wf()

    # fmt: off
    workflow.connect([
        (inputnode, load_bmat, [('in_file', 'in_file')]),
        (inputnode, dwi_report_wf, [
            ('in_file', 'inputnode.name_source'),
        ]),
        (inputnode, iqms_wf, [
            ('in_file', 'inputnode.in_file'),
            ('metadata', 'inputnode.metadata'),
            ('entities', 'inputnode.entities'),
        ]),
        (inputnode, sanitize, [('in_file', 'in_file')]),
        (sanitize, dwi_ref, [('out_file', 'in_file')]),
        (sanitize, sp_mask, [('out_file', 'in_file')]),
        (sanitize, piesno, [('out_file', 'in_file')]),
        (shells, dwi_ref, [(('b_masks', _first), 't_mask')]),
        (shells, sp_mask, [('b_masks', 'b_masks')]),
        (load_bmat, shells, [('out_bval_file', 'in_bvals')]),
        (sanitize, drift, [('out_file', 'full_epi')]),
        (shells, get_lowb, [(('b_indices', _first), 'indices')]),
        (sanitize, get_lowb, [('out_file', 'in_file')]),
        (load_bmat, drift, [('out_bval_file', 'bval_file')]),
        (get_lowb, hmc_b0, [('out_file', 'in_file')]),
        (dwi_ref, hmc_b0, [('out_file', 'basefile')]),
        (hmc_b0, drift, [('out_file', 'in_file')]),
        (shells, drift, [(('b_indices', _first), 'b0_ixs')]),
        (dwi_ref, dmri_bmsk, [('out_file', 'inputnode.in_files')]),
        (dmri_bmsk, sp_mask, [('outputnode.out_mask', 'brain_mask')]),
        (dmri_bmsk, drift, [('outputnode.out_mask', 'brainmask_file')]),
        (drift, hmcwf, [('out_full_file', 'inputnode.in_file')]),
        (load_bmat, hmcwf, [('out_bvec_file', 'inputnode.in_bvec')]),
        (drift, averages, [('out_full_file', 'in_file')]),
        (drift, stddev, [('out_full_file', 'in_file')]),
        (shells, averages, [('b_masks', 'in_weights')]),
        (averages, hmcwf, [(('out_file', _first), 'inputnode.reference')]),
        (shells, stddev, [('b_masks', 'in_weights')]),
        (shells, dwimodel, [('out_data', 'bvals'),
                            ('n_shells', 'n_shells')]),
        (load_bmat, dwimodel, [('out_bvec_file', 'bvec_file')]),
        (drift, dwidenoise, [('out_full_file', 'in_file')]),
        (dmri_bmsk, dwidenoise, [('outputnode.out_mask', 'mask')]),
        (dwidenoise, dwimodel, [('out_file', 'in_file')]),
        (dmri_bmsk, dwimodel, [('outputnode.out_mask', 'brain_mask')]),
        (load_bmat, get_hmc_shells, [('out_bvec_file', 'in_bvec_file')]),
        (shells, get_hmc_shells, [('b_indices', 'indices')]),
        (hmcwf, get_hmc_shells, [('outputnode.out_file', 'in_file')]),
        (dwimodel, cc_mask, [('out_fa', 'in_fa'),
                             ('out_cfa', 'in_cfa')]),
        (load_bmat, iqms_wf, [
            ('out_bval_file', 'inputnode.b_values_file'),
            ('qspace_neighbors', 'inputnode.qspace_neighbors'),
        ]),
        (averages, iqms_wf, [(('out_file', _first), 'inputnode.in_b0')]),
        (sp_mask, iqms_wf, [('out_mask', 'inputnode.spikes_mask')]),
        (piesno, iqms_wf, [('sigma', 'inputnode.piesno_sigma')]),
        (hmcwf, iqms_wf, [('outputnode.out_fd', 'inputnode.framewise_displacement'),
                          ('outputnode.out_bvec', 'inputnode.in_bvec_rotated'),
                          ('outputnode.out_bvec_diff', 'inputnode.in_bvec_diff')]),
        (dwimodel, iqms_wf, [('out_fa', 'inputnode.in_fa'),
                             ('out_cfa', 'inputnode.in_cfa'),
                             ('out_fa_nans', 'inputnode.in_fa_nans'),
                             ('out_fa_degenerate', 'inputnode.in_fa_degenerate'),
                             ('out_md', 'inputnode.in_md')]),
        (dmri_bmsk, iqms_wf, [('outputnode.out_mask', 'inputnode.brain_mask')]),
        (cc_mask, iqms_wf, [('out_mask', 'inputnode.cc_mask'),
                            ('wm_finalmask', 'inputnode.wm_mask')]),
        (shells, iqms_wf, [('n_shells', 'inputnode.n_shells'),
                           ('b_values', 'inputnode.b_values_shells')]),
        (get_hmc_shells, iqms_wf, [('out_file', 'inputnode.in_shells'),
                                   ('out_bvec', 'inputnode.in_bvec')]),
        (dwidenoise, iqms_wf, [('noise', 'inputnode.in_noise')]),
        (dwi_ref, spatial_norm, [('out_file', 'inputnode.epi_mean')]),
        (dmri_bmsk, spatial_norm, [('outputnode.out_mask', 'inputnode.epi_mask')]),
        (iqms_wf, dwi_report_wf, [('outputnode.noise_floor', 'inputnode.noise_floor')]),
        (shells, dwi_report_wf, [('b_dict', 'inputnode.in_bdict')]),
        (dmri_bmsk, dwi_report_wf, [('outputnode.out_mask', 'inputnode.brain_mask')]),
        (shells, dwi_report_wf, [('b_values', 'inputnode.in_shells')]),
        (averages, dwi_report_wf, [('out_file', 'inputnode.in_avgmap')]),
        (stddev, dwi_report_wf, [('out_file', 'inputnode.in_stdmap')]),
        (drift, dwi_report_wf, [('out_full_file', 'inputnode.in_epi')]),
        (dwimodel, dwi_report_wf, [('out_fa', 'inputnode.in_fa'),
                                   ('out_md', 'inputnode.in_md')]),
        (spatial_norm, dwi_report_wf, [('outputnode.epi_parc', 'inputnode.in_parcellation')]),
    ])
    # fmt: on
    return workflow


def compute_iqms(name='ComputeIQMs'):
    """
    Initialize the workflow that actually computes the IQMs.

    .. workflow::

        from mriqc.workflows.diffusion.base import compute_iqms
        from mriqc.testing import mock_config
        with mock_config():
            wf = compute_iqms()

    """

    from mriqc.interfaces import IQMFileSink
    from mriqc.interfaces.diffusion import DiffusionQC
    from mriqc.interfaces.reports import AddProvenance

    # from mriqc.workflows.utils import _tofloat, get_fwhmx

    workflow = pe.Workflow(name=name)
    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                'in_file',
                'metadata',
                'entities',
                'in_shells',
                'n_shells',
                'b_values_file',
                'b_values_shells',
                'in_bvec',
                'in_bvec_rotated',
                'in_bvec_diff',
                'in_b0',
                'in_fa',
                'in_cfa',
                'in_fa_nans',
                'in_fa_degenerate',
                'in_md',
                'in_noise',
                'brain_mask',
                'wm_mask',
                'cc_mask',
                'spikes_mask',
                'framewise_displacement',
                'qspace_neighbors',
                'piesno_sigma',
            ]
        ),
        name='inputnode',
    )
    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                'out_file',
                'noise_floor',
            ]
        ),
        name='outputnode',
    )

    estimate_sigma = pe.Node(
        niu.Function(function=_estimate_sigma),
        name='estimate_sigma',
    )

    measures = pe.Node(DiffusionQC(), name='measures')

    addprov = pe.Node(
        AddProvenance(modality='dwi'),
        name='provenance',
        run_without_submitting=True,
    )

    # Save to JSON file
    datasink = pe.Node(
        IQMFileSink(
            modality='dwi',
            out_dir=str(config.execution.output_dir),
            dataset=config.execution.dsname,
        ),
        name='datasink',
        run_without_submitting=True,
    )

    # fmt: off
    workflow.connect([
        (inputnode, datasink, [('in_file', 'in_file'),
                               ('entities', 'entities'),
                               (('metadata', _filter_metadata), 'metadata'),
                               ('n_shells', 'NumberOfShells'),
                               ('b_values_shells', 'bValuesEstimation'),
                               (('b_values_file', _bvals_report), 'bValues')]),
        (inputnode, measures, [('in_file', 'in_file'),
                               ('b_values_file', 'in_bval_file'),
                               ('b_values_shells', 'in_shells_bval'),
                               ('in_shells', 'in_shells'),
                               ('in_bvec', 'in_bvec'),
                               ('in_bvec_rotated', 'in_bvec_rotated'),
                               ('in_bvec_diff', 'in_bvec_diff'),
                               ('in_b0', 'in_b0'),
                               ('brain_mask', 'brain_mask'),
                               ('wm_mask', 'wm_mask'),
                               ('cc_mask', 'cc_mask'),
                               ('spikes_mask', 'spikes_mask'),
                               ('in_fa', 'in_fa'),
                               ('in_md', 'in_md'),
                               ('in_cfa', 'in_cfa'),
                               ('in_fa_nans', 'in_fa_nans'),
                               ('in_fa_degenerate', 'in_fa_degenerate'),
                               ('framewise_displacement', 'in_fd'),
                               ('qspace_neighbors', 'qspace_neighbors'),
                               ('piesno_sigma', 'piesno_sigma')]),
        (inputnode, addprov, [('in_file', 'in_file')]),
        (addprov, datasink, [('out_prov', 'provenance')]),
        (datasink, outputnode, [('out_file', 'out_file')]),
        (measures, datasink, [('out_qc', 'root')]),
        (inputnode, estimate_sigma, [('in_noise', 'in_file'),
                                     ('brain_mask', 'mask')]),
        (estimate_sigma, measures, [('out', 'noise_floor')]),
        (estimate_sigma, outputnode, [('out', 'noise_floor')]),
    ])
    # fmt: on
    return workflow


def hmc_workflow(name='dMRI_HMC'):
    """
    Create a :abbr:`HMC (head motion correction)` workflow for dMRI.

    .. workflow::

        from mriqc.workflows.diffusion.base import hmc
        from mriqc.testing import mock_config
        with mock_config():
            wf = hmc()

    """
    from nipype.algorithms.confounds import FramewiseDisplacement
    from nipype.interfaces.afni import Volreg

    from mriqc.interfaces.diffusion import RotateVectors

    workflow = pe.Workflow(name=name)

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                'in_file',
                'reference',
                'in_bvec',
            ]
        ),
        name='inputnode',
    )
    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                'out_file',
                'out_fd',
                'out_bvec',
                'out_bvec_diff',
            ]
        ),
        name='outputnode',
    )

    # calculate hmc parameters
    hmc = pe.Node(
        Volreg(args='-Fourier -twopass', zpad=4, outputtype='NIFTI_GZ'),
        name='motion_correct',
        mem_gb=3.0,
        n_procs=config.nipype.nprocs,
    )

    bvec_rot = pe.Node(RotateVectors(), name='bvec_rot')

    # Compute the frame-wise displacement
    fdnode = pe.Node(
        FramewiseDisplacement(
            normalize=False,
            parameter_source='AFNI',
            radius=config.workflow.fd_radius,
        ),
        name='ComputeFD',
    )

    # fmt: off
    workflow.connect([
        (inputnode, hmc, [('in_file', 'in_file'),
                          ('reference', 'basefile')]),
        (inputnode, bvec_rot, [('in_bvec', 'in_file'),
                               ('reference', 'reference')]),
        (hmc, outputnode, [('out_file', 'out_file')]),
        (hmc, fdnode, [('oned_file', 'in_file')]),
        (hmc, bvec_rot, [('oned_matrix_save', 'transforms')]),
        (fdnode, outputnode, [('out_file', 'out_fd')]),
        (bvec_rot, outputnode, [('out_bvec', 'out_bvec'),
                                ('out_diff', 'out_bvec_diff')]),
    ])
    # fmt: on
    return workflow


def epi_mni_align(name='SpatialNormalization'):
    """
    Estimate the transform that maps the EPI space into MNI152NLin2009cAsym.

    The input epi_mean is the averaged and brain-masked EPI timeseries

    Returns the EPI mean resampled in MNI space (for checking out registration) and
    the associated "lobe" parcellation in EPI space.

    .. workflow::

        from mriqc.workflows.diffusion.base import epi_mni_align
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


def _mean(inlist):
    from numpy import mean

    return mean(inlist)


def _parse_tqual(in_file):
    from numpy import mean

    with open(in_file) as fin:
        lines = fin.readlines()
    return mean([float(line.strip()) for line in lines if not line.startswith('++')])


def _parse_tout(in_file):
    from numpy import loadtxt

    data = loadtxt(in_file)  # pylint: disable=no-member
    return data.mean()


def _tolist(value):
    return [value]


def _get_bvals(bmatrix):
    from numpy import squeeze

    return squeeze(bmatrix[:, -1]).tolist()


def _first(inlist):
    if isinstance(inlist, (list, tuple)):
        return inlist[0]

    return inlist


def _all_but_first(inlist):
    if isinstance(inlist, (list, tuple)):
        return inlist[1:]

    return inlist


def _estimate_sigma(in_file, mask):
    import nibabel as nb
    from numpy import median

    msk = nb.load(mask).get_fdata() > 0.5
    return round(
        float(median(nb.load(in_file).get_fdata()[msk])),
        6,
    )


def _bvals_report(in_file):
    import numpy as np

    bvals = [round(float(val), 2) for val in np.unique(np.round(np.loadtxt(in_file), 2))]

    if len(bvals) > 10:
        return 'Likely DSI'

    return bvals


def _filter_metadata(
    in_dict,
    keys=(
        'global',
        'dcmmeta_affine',
        'dcmmeta_reorient_transform',
        'dcmmeta_shape',
        'dcmmeta_slice_dim',
        'dcmmeta_version',
        'time',
    ),
):
    """Drop large and partially redundant objects generated by dcm2niix."""

    for key in keys:
        in_dict.pop(key, None)

    return in_dict
