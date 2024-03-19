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
from mriqc.interfaces.datalad import DataladIdentityInterface
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

    from mriqc.interfaces.diffusion import (
        CorrectSignalDrift,
        DipyDTI,
        ExtractB0,
        FilterShells,
        NumberOfShells,
        ReadDWIMetadata,
        WeightedStat,
    )
    from mriqc.messages import BUILDING_WORKFLOW
    from mriqc.workflows.shared import synthstrip_wf as dmri_bmsk_workflow

    workflow = pe.Workflow(name=name)

    mem_gb = config.workflow.biggest_file_gb

    dataset = config.workflow.inputs.get('dwi', [])

    message = BUILDING_WORKFLOW.format(
        modality='diffusion',
        detail=(
            f'for {len(dataset)} NIfTI files.'
            if len(dataset) > 2
            else f"({' and '.join('<%s>' % v for v in dataset)})."
        ),
    )
    config.loggers.workflow.info(message)

    # Define workflow, inputs and outputs
    # 0. Get data, put it in RAS orientation
    inputnode = pe.Node(niu.IdentityInterface(fields=['in_file']), name='inputnode')
    inputnode.iterables = [('in_file', dataset)]

    datalad_get = pe.Node(
        DataladIdentityInterface(fields=['in_file'], dataset_path=config.execution.bids_dir),
        name='datalad_get',
    )

    outputnode = pe.Node(
        niu.IdentityInterface(fields=['qc', 'mosaic', 'out_group', 'out_dvars', 'out_fd']),
        name='outputnode',
    )

    sanitize = pe.Node(
        SanitizeImage(
            n_volumes_to_discard=0,
            max_32bit=config.execution.float32,
        ),
        name='sanitize',
        mem_gb=mem_gb * 4.0,
    )

    # Workflow --------------------------------------------------------

    # 1. Read metadata & bvec/bval, estimate number of shells, extract and split B0s
    meta = pe.Node(ReadDWIMetadata(index_db=config.execution.bids_database_dir), name='metadata')
    shells = pe.Node(NumberOfShells(), name='shells')
    get_shells = pe.MapNode(ExtractB0(), name='get_shells', iterfield=['b0_ixs'])
    hmc_shells = pe.MapNode(
        Volreg(args='-Fourier -twopass', zpad=4, outputtype='NIFTI_GZ'),
        name='hmc_shells',
        mem_gb=mem_gb * 2.5,
        iterfield=['in_file'],
    )

    hmc_b0 = pe.Node(
        Volreg(args='-Fourier -twopass', zpad=4, outputtype='NIFTI_GZ'),
        name='hmc_b0',
        mem_gb=mem_gb * 2.5,
    )

    drift = pe.Node(CorrectSignalDrift(), name='drift')

    # 2. Generate B0 reference
    dwi_reference_wf = init_dmriref_wf(name='dwi_reference_wf')

    # 3. Calculate brainmask
    dmri_bmsk = dmri_bmsk_workflow(omp_nthreads=config.nipype.omp_nthreads)

    # 4. HMC: head motion correct
    hmcwf = hmc_workflow()

    # 5. Split shells and compute some stats
    averages = pe.MapNode(
        WeightedStat(),
        name='averages',
        mem_gb=mem_gb * 1.5,
        iterfield=['in_weights'],
    )
    stddev = pe.MapNode(
        WeightedStat(stat='std'),
        name='stddev',
        mem_gb=mem_gb * 1.5,
        iterfield=['in_weights'],
    )

    # 6. Fit DTI model
    dti_filter = pe.Node(FilterShells(), name='dti_filter')
    dwidenoise = pe.Node(
        DWIDenoise(
            noise='noisemap.nii.gz',
            nthreads=config.nipype.omp_nthreads,
        ),
        name='dwidenoise',
        nprocs=config.nipype.omp_nthreads,
    )
    dti = pe.Node(
        DipyDTI(free_water_model=False),
        name='dti',
    )

    # 7. EPI to MNI registration
    ema = epi_mni_align()

    # 8. Compute IQMs
    iqmswf = compute_iqms()

    # 9. Generate outputs
    dwi_report_wf = init_dwi_report_wf()

    # fmt: off
    workflow.connect([
        (inputnode, datalad_get, [('in_file', 'in_file')]),
        (inputnode, meta, [('in_file', 'in_file')]),
        (inputnode, dwi_report_wf, [
            ('in_file', 'inputnode.name_source'),
        ]),
        (datalad_get, iqmswf, [('in_file', 'inputnode.in_file')]),
        (datalad_get, sanitize, [('in_file', 'in_file')]),
        (sanitize, dwi_reference_wf, [('out_file', 'inputnode.in_file')]),
        (shells, dwi_reference_wf, [(('b_masks', _first), 'inputnode.t_mask')]),
        (meta, shells, [('out_bval_file', 'in_bvals')]),
        (sanitize, drift, [('out_file', 'full_epi')]),
        (shells, get_shells, [('b_indices', 'b0_ixs')]),
        (sanitize, get_shells, [('out_file', 'in_file')]),
        (meta, drift, [('out_bval_file', 'bval_file')]),
        (get_shells, hmc_shells, [(('out_file', _all_but_first), 'in_file')]),
        (get_shells, hmc_b0, [(('out_file', _first), 'in_file')]),
        (dwi_reference_wf, hmc_b0, [('outputnode.ref_file', 'basefile')]),
        (hmc_b0, drift, [('out_file', 'in_file')]),
        (shells, drift, [(('b_indices', _first), 'b0_ixs')]),
        (dwi_reference_wf, dmri_bmsk, [('outputnode.ref_file', 'inputnode.in_files')]),
        (dwi_reference_wf, ema, [('outputnode.ref_file', 'inputnode.epi_mean')]),
        (dmri_bmsk, drift, [('outputnode.out_mask', 'brainmask_file')]),
        (dmri_bmsk, ema, [('outputnode.out_mask', 'inputnode.epi_mask')]),
        (drift, hmcwf, [('out_full_file', 'inputnode.reference')]),
        (drift, averages, [('out_full_file', 'in_file')]),
        (drift, stddev, [('out_full_file', 'in_file')]),
        (shells, averages, [('b_masks', 'in_weights')]),
        (shells, stddev, [('b_masks', 'in_weights')]),
        (shells, dti_filter, [('out_data', 'bvals')]),
        (meta, dti_filter, [('out_bvec_file', 'bvec_file')]),
        (drift, dti_filter, [('out_full_file', 'in_file')]),
        (dti_filter, dti, [('out_bvals', 'bvals')]),
        (dti_filter, dti, [('out_bvec_file', 'bvec_file')]),
        (dti_filter, dwidenoise, [('out_file', 'in_file')]),
        (dmri_bmsk, dwidenoise, [('outputnode.out_mask', 'mask')]),
        (dwidenoise, dti, [('out_file', 'in_file')]),
        (dmri_bmsk, dti, [('outputnode.out_mask', 'brainmask')]),
        (hmcwf, outputnode, [('outputnode.out_fd', 'out_fd')]),
        (shells, iqmswf, [('n_shells', 'inputnode.n_shells'),
                          ('b_values', 'inputnode.b_values')]),
        (dwidenoise, dwi_report_wf, [('noise', 'inputnode.in_noise')]),
        (shells, dwi_report_wf, [('b_dict', 'inputnode.in_bdict')]),
        (dmri_bmsk, dwi_report_wf, [('outputnode.out_mask', 'inputnode.brainmask')]),
        (shells, dwi_report_wf, [('b_values', 'inputnode.in_shells')]),
        (averages, dwi_report_wf, [('out_file', 'inputnode.in_avgmap')]),
        (stddev, dwi_report_wf, [('out_file', 'inputnode.in_stdmap')]),
        (drift, dwi_report_wf, [('out_full_file', 'inputnode.in_epi')]),
        (dti, dwi_report_wf, [('out_fa', 'inputnode.in_fa'),
                              ('out_md', 'inputnode.in_md')]),
        (ema, dwi_report_wf, [('outputnode.epi_parc', 'inputnode.in_parcellation')]),
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
    from niworkflows.interfaces.bids import ReadSidecarJSON

    from mriqc.interfaces import IQMFileSink
    from mriqc.interfaces.reports import AddProvenance

    # from mriqc.workflows.utils import _tofloat, get_fwhmx
    # mem_gb = config.workflow.biggest_file_gb

    workflow = pe.Workflow(name=name)
    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                'in_file',
                'n_shells',
                'b_values',
            ]
        ),
        name='inputnode',
    )
    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                'out_file',
                'meta_sidecar',
            ]
        ),
        name='outputnode',
    )

    meta = pe.Node(ReadSidecarJSON(index_db=config.execution.bids_database_dir), name='metadata')

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
                               ('n_shells', 'NumberOfShells'),
                               ('b_values', 'b-values')]),
        (inputnode, meta, [('in_file', 'in_file')]),
        (inputnode, addprov, [('in_file', 'in_file')]),
        (addprov, datasink, [('out_prov', 'provenance')]),
        (meta, datasink, [('subject', 'subject_id'),
                          ('session', 'session_id'),
                          ('task', 'task_id'),
                          ('acquisition', 'acq_id'),
                          ('reconstruction', 'rec_id'),
                          ('run', 'run_id'),
                          ('out_dict', 'metadata')]),
        (datasink, outputnode, [('out_file', 'out_file')]),
        (meta, outputnode, [('out_dict', 'meta_sidecar')]),
    ])
    # fmt: on

    # Set FD threshold
    # inputnode.inputs.fd_thres = config.workflow.fd_thres

    # # AFNI quality measures
    # fwhm_interface = get_fwhmx()
    # fwhm = pe.Node(fwhm_interface, name="smoothness")
    # # fwhm.inputs.acf = True  # add when AFNI >= 16
    # measures = pe.Node(FunctionalQC(), name="measures", mem_gb=mem_gb * 3)

    # # fmt: off
    # workflow.connect([
    #     (inputnode, measures, [("epi_mean", "in_epi"),
    #                            ("brainmask", "in_mask"),
    #                            ("hmc_epi", "in_hmc"),
    #                            ("hmc_fd", "in_fd"),
    #                            ("fd_thres", "fd_thres"),
    #                            ("in_tsnr", "in_tsnr")]),
    #     (inputnode, fwhm, [("epi_mean", "in_file"),
    #                        ("brainmask", "mask")]),
    #     (fwhm, measures, [(("fwhm", _tofloat), "in_fwhm")]),
    #     (measures, datasink, [("out_qc", "root")]),
    # ])
    # # fmt: on
    return workflow


def init_dmriref_wf(
    in_file=None,
    name='init_dmriref_wf',
):
    """
    Build a workflow that generates reference images for a dMRI series.

    The raw reference image is the target of :abbr:`HMC (head motion correction)`, and a
    contrast-enhanced reference is the subject of distortion correction, as well as
    boundary-based registration to T1w and template spaces.

    This workflow assumes only one dMRI file has been passed.

    Workflow Graph
        .. workflow::
            :graph2use: orig
            :simple_form: yes

            from mriqc.workflows.diffusion.base import init_dmriref_wf
            wf = init_dmriref_wf()

    Parameters
    ----------
    in_file : :obj:`str`
        dMRI series NIfTI file
    ------
    in_file : str
        series NIfTI file

    Outputs
    -------
    in_file : str
        Validated DWI series NIfTI file
    ref_file : str
        Reference image to which DWI series is motion corrected
    """
    from niworkflows.interfaces.header import ValidateImage
    from niworkflows.interfaces.images import RobustAverage

    workflow = pe.Workflow(name=name)
    inputnode = pe.Node(niu.IdentityInterface(fields=['in_file', 't_mask']), name='inputnode')
    outputnode = pe.Node(
        niu.IdentityInterface(fields=['in_file', 'ref_file', 'validation_report']),
        name='outputnode',
    )

    # Simplify manually setting input image
    if in_file is not None:
        inputnode.inputs.in_file = in_file

    val_bold = pe.Node(
        ValidateImage(),
        name='val_bold',
        mem_gb=DEFAULT_MEMORY_MIN_GB,
    )

    gen_avg = pe.Node(RobustAverage(mc_method=None), name='gen_avg', mem_gb=1)
    # fmt: off
    workflow.connect([
        (inputnode, val_bold, [('in_file', 'in_file')]),
        (inputnode, gen_avg, [('t_mask', 't_mask')]),
        (val_bold, gen_avg, [('out_file', 'in_file')]),
        (gen_avg, outputnode, [('out_file', 'ref_file')]),
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

    mem_gb = config.workflow.biggest_file_gb

    workflow = pe.Workflow(name=name)

    inputnode = pe.Node(niu.IdentityInterface(fields=['in_file', 'reference']), name='inputnode')
    outputnode = pe.Node(niu.IdentityInterface(fields=['out_file', 'out_fd']), name='outputnode')

    # calculate hmc parameters
    hmc = pe.Node(
        Volreg(args='-Fourier -twopass', zpad=4, outputtype='NIFTI_GZ'),
        name='motion_correct',
        mem_gb=mem_gb * 2.5,
    )

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
        (hmc, outputnode, [('out_file', 'out_file')]),
        (hmc, fdnode, [('oned_file', 'in_file')]),
        (fdnode, outputnode, [('out_file', 'out_fd')]),
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
    import numpy as np

    return np.mean(inlist)


def _parse_tqual(in_file):
    import numpy as np

    with open(in_file) as fin:
        lines = fin.readlines()
    return np.mean([float(line.strip()) for line in lines if not line.startswith('++')])


def _parse_tout(in_file):
    import numpy as np

    data = np.loadtxt(in_file)  # pylint: disable=no-member
    return data.mean()


def _tolist(value):
    return [value]


def _get_bvals(bmatrix):
    import numpy

    return numpy.squeeze(bmatrix[:, -1]).tolist()


def _first(inlist):
    if isinstance(inlist, (list, tuple)):
        return inlist[0]

    return inlist


def _all_but_first(inlist):
    if isinstance(inlist, (list, tuple)):
        return inlist[1:]

    return inlist
