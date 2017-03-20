#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# @Author: oesteban
# @Date:   2016-01-05 16:15:08
# @Email:  code@oscaresteban.es
"""
=======================
The functional workflow
=======================

The functional workflow follows the following steps:

#. Conform (reorientations, revise data types) input data, read
   associated metadata and discard non-steady state frames
   (:py:func:`mriqc.utils.misc.reorient_and_discard_non_steady`).
#. :abbr:`HMC (head-motion correction)` based on ``3dvolreg`` from
   AFNI -- :py:func:`hmc_afni`.
#. Skull-stripping of the time-series (AFNI) --
   :py:func:`fmri_bmsk_workflow`.
#. Calculate mean time-series, and :abbr:`tSNR (temporal SNR)`.
#. Spatial Normalization to MNI (ANTs) -- :py:func:`epi_mni_align`
#. Extraction of IQMs -- :py:func:`compute_iqms`.
#. Individual-reports generation -- :py:func:`individual_reports`.

This workflow is orchestrated by :py:func:`fmri_qc_workflow`.

"""
from __future__ import print_function, division, absolute_import, unicode_literals
import os
import os.path as op

from nipype import logging
from nipype.pipeline import engine as pe
from nipype.algorithms import confounds as nac
from nipype.interfaces import io as nio
from nipype.interfaces import utility as niu
from nipype.interfaces import fsl
from nipype.interfaces import afni

from mriqc import DEFAULTS
from mriqc.workflows.utils import fwhm_dict, slice_wise_fft
from mriqc.interfaces import ReadSidecarJSON, FunctionalQC, Spikes, IQMFileSink
from mriqc.utils.misc import check_folder, reorient_and_discard_non_steady


DEFAULT_FD_RADIUS = 50.
WFLOGGER = logging.getLogger('workflow')

def fmri_qc_workflow(dataset, settings, name='funcMRIQC'):
    """
    The fMRI qc workflow

    .. workflow::

      import os.path as op
      from mriqc.workflows.functional import fmri_qc_workflow
      datadir = op.abspath('data')
      wf = fmri_qc_workflow([op.join(datadir, 'sub-001/func/sub-001_task-rest_bold.nii.gz')],
                            settings={'bids_dir': datadir,
                                      'output_dir': op.abspath('out')})


    """

    workflow = pe.Workflow(name=name)

    # Define workflow, inputs and outputs
    # 0. Get data, put it in RAS orientation
    inputnode = pe.Node(niu.IdentityInterface(fields=['in_file']), name='inputnode')
    WFLOGGER.info('Building fMRI QC workflow, datasets list: %s',
                  sorted([d.replace(settings['bids_dir'] + '/', '') for d in dataset]))
    inputnode.iterables = [('in_file', dataset)]


    meta = pe.Node(ReadSidecarJSON(), name='metadata')

    outputnode = pe.Node(niu.IdentityInterface(
        fields=['qc', 'mosaic', 'out_group', 'out_dvars',
                'out_fd']), name='outputnode')


    reorient_and_discard = pe.Node(niu.Function(input_names=['in_file', 'float32'],
                                                output_names=['exclude_index',
                                                              'out_file'],
                                                function=reorient_and_discard_non_steady),
                                   name='reorient_and_discard')
    reorient_and_discard.inputs.float32 = settings.get("float32", DEFAULTS['float32'])

    # Workflow --------------------------------------------------------

    # 1. HMC: head motion correct
    if settings.get('hmc_fsl', False):
        assert not settings.get('hmc_afni', False)
        hmcwf = hmc_mcflirt()
    else:
        assert settings.get('hmc_afni', True)
        hmcwf = hmc_afni(st_correct=settings.get('correct_slice_timing', False),
                         despike=settings.get('despike', False),
                         deoblique=settings.get('deoblique', False),
                         start_idx=settings.get('start_idx', None),
                         stop_idx=settings.get('stop_idx', None))

    # Set HMC settings
    hmcwf.inputs.inputnode.fd_radius = settings.get('fd_radius', DEFAULT_FD_RADIUS)


    mean = pe.Node(afni.TStat(                   # 2. Compute mean fmri
        options='-mean', outputtype='NIFTI_GZ'), name='mean')
    bmw = fmri_bmsk_workflow(                   # 3. Compute brain mask
        use_bet=settings.get('use_bet', False))

    # EPI to MNI registration
    ema = epi_mni_align(ants_nthreads=settings.get('ants_nthreads', DEFAULTS['ants_nthreads']),
                        testing=settings.get('testing', False))

    # Compute TSNR using nipype implementation
    tsnr = pe.Node(nac.TSNR(), name='compute_tsnr')

    # 7. Compute IQMs
    iqmswf = compute_iqms(settings)
    # Reports
    repwf = individual_reports(settings)

    workflow.connect([
        (inputnode, meta, [('in_file', 'in_file')]),
        (inputnode, reorient_and_discard, [('in_file', 'in_file')]),
        (reorient_and_discard, hmcwf, [('out_file', 'inputnode.in_file')]),
        (hmcwf, bmw, [('outputnode.out_file', 'inputnode.in_file')]),
        (hmcwf, mean, [('outputnode.out_file', 'in_file')]),
        (hmcwf, tsnr, [('outputnode.out_file', 'in_file')]),
        (mean, ema, [('out_file', 'inputnode.epi_mean')]),
        (bmw, ema, [('outputnode.out_file', 'inputnode.epi_mask')]),
        (meta, iqmswf, [('subject_id', 'inputnode.subject_id'),
                        ('session_id', 'inputnode.session_id'),
                        ('task_id', 'inputnode.task_id'),
                        ('acq_id', 'inputnode.acq_id'),
                        ('rec_id', 'inputnode.rec_id'),
                        ('run_id', 'inputnode.run_id'),
                        ('out_dict', 'inputnode.metadata')]),
        (reorient_and_discard, iqmswf, [('out_file', 'inputnode.orig')]),
        (mean, iqmswf, [('out_file', 'inputnode.epi_mean')]),
        (hmcwf, iqmswf, [('outputnode.out_file', 'inputnode.hmc_epi'),
                         ('outputnode.out_fd', 'inputnode.hmc_fd')]),
        (bmw, iqmswf, [('outputnode.out_file', 'inputnode.brainmask')]),
        (tsnr, iqmswf, [('tsnr_file', 'inputnode.in_tsnr')]),
        (reorient_and_discard, repwf, [('out_file', 'inputnode.orig')]),
        (mean, repwf, [('out_file', 'inputnode.epi_mean')]),
        (tsnr, repwf, [('stddev_file', 'inputnode.in_stddev')]),
        (bmw, repwf, [('outputnode.out_file', 'inputnode.brainmask')]),
        (hmcwf, repwf, [('outputnode.out_fd', 'inputnode.hmc_fd'),
                        ('outputnode.out_file', 'inputnode.hmc_epi')]),
        (ema, repwf, [('outputnode.epi_parc', 'inputnode.epi_parc'),
                      ('outputnode.report', 'inputnode.mni_report')]),
        (reorient_and_discard, repwf, [('exclude_index', 'inputnode.exclude_index')]),
        (iqmswf, repwf, [('outputnode.out_file', 'inputnode.in_iqms'),
                         ('outputnode.out_dvars', 'inputnode.in_dvars'),
                         ('outputnode.outliers', 'inputnode.outliers')]),
        (hmcwf, outputnode, [('outputnode.out_fd', 'out_fd')]),
    ])

    if settings.get('fft_spikes_detector', False):
        workflow.connect([
            (iqmswf, repwf, [('outputnode.out_spikes', 'inputnode.in_spikes'),
                             ('outputnode.out_fft', 'inputnode.in_fft')]),
        ])

    return workflow

def compute_iqms(settings, name='ComputeIQMs'):
    """
    Workflow that actually computes the IQMs

    .. workflow::

      from mriqc.workflows.functional import compute_iqms
      wf = compute_iqms(settings={'output_dir': 'out'})


    """
    workflow = pe.Workflow(name=name)
    inputnode = pe.Node(niu.IdentityInterface(fields=[
        'subject_id', 'session_id', 'task_id', 'acq_id', 'rec_id', 'run_id', 'orig',
        'epi_mean', 'brainmask', 'hmc_epi', 'hmc_fd', 'in_tsnr', 'metadata']), name='inputnode')
    outputnode = pe.Node(niu.IdentityInterface(
        fields=['out_file', 'out_dvars', 'outliers', 'out_spikes', 'out_fft']),
                         name='outputnode')

    deriv_dir = check_folder(op.abspath(op.join(settings['output_dir'], 'derivatives')))

    # Compute DVARS
    dvnode = pe.Node(nac.ComputeDVARS(save_plot=False, save_all=True), name='ComputeDVARS')

    # AFNI quality measures
    fwhm = pe.Node(afni.FWHMx(combine=True, detrend=True), name='smoothness')
    # fwhm.inputs.acf = True  # add when AFNI >= 16
    outliers = pe.Node(afni.OutlierCount(fraction=True, out_file='ouliers.out'),
                       name='outliers')
    quality = pe.Node(afni.QualityIndex(automask=True), out_file='quality.out',
                      name='quality')

    measures = pe.Node(FunctionalQC(), name='measures')

    workflow.connect([
        (inputnode, dvnode, [('hmc_epi', 'in_file'),
                             ('brainmask', 'in_mask')]),
        (inputnode, measures, [('epi_mean', 'in_epi'),
                               ('brainmask', 'in_mask'),
                               ('hmc_epi', 'in_hmc'),
                               ('hmc_fd', 'in_fd'),
                               ('in_tsnr', 'in_tsnr')]),
        (inputnode, fwhm, [('epi_mean', 'in_file'),
                           ('brainmask', 'mask')]),
        (inputnode, quality, [('hmc_epi', 'in_file')]),
        (inputnode, outliers, [('hmc_epi', 'in_file'),
                               ('brainmask', 'mask')]),
        (dvnode, measures, [('out_all', 'in_dvars')]),
        (dvnode, outputnode, [('out_all', 'out_dvars')]),
        (outliers, outputnode, [('out_file', 'outliers')])
    ])

    # Save to JSON file
    datasink = pe.Node(IQMFileSink(
        modality='bold', out_dir=deriv_dir), name='datasink')

    workflow.connect([
        (inputnode, datasink, [('subject_id', 'subject_id'),
                               ('session_id', 'session_id'),
                               ('task_id', 'task_id'),
                               ('acq_id', 'acq_id'),
                               ('rec_id', 'rec_id'),
                               ('run_id', 'run_id'),
                               ('metadata', 'metadata')]),
        (outliers, datasink, [(('out_file', _parse_tout), 'aor')]),
        (quality, datasink, [(('out_file', _parse_tqual), 'aqi')]),
        (measures, datasink, [('out_qc', 'root')]),
        (fwhm, datasink, [(('fwhm', fwhm_dict), 'root0')]),
        (datasink, outputnode, [('out_file', 'out_file')])
    ])

    if settings.get('fft_spikes_detector', False):
        # FFT spikes finder
        spikes_fft = pe.Node(niu.Function(
            input_names=['in_file'],
            output_names=['n_spikes', 'out_spikes', 'out_fft'],
            function=slice_wise_fft), name='SpikesFinderFFT')

        workflow.connect([
            (inputnode, spikes_fft, [('orig', 'in_file')]),
            (spikes_fft, outputnode, [('out_spikes', 'out_spikes'),
                                      ('out_fft', 'out_fft')]),
            (spikes_fft, datasink, [('n_spikes', 'spikes_num')])
        ])


    return workflow


def individual_reports(settings, name='ReportsWorkflow'):
    """
    Encapsulates nodes writing plots

    .. workflow::

      from mriqc.workflows.functional import individual_reports
      wf = individual_reports(settings={'output_dir': 'out'})

    """
    from mriqc.interfaces import PlotMosaic, PlotSpikes
    from mriqc.reports import individual_html

    verbose = settings.get('verbose_reports', False)
    pages = 4
    extra_pages = 0
    if verbose:
        extra_pages = 4

    workflow = pe.Workflow(name=name)
    inputnode = pe.Node(niu.IdentityInterface(fields=[
        'in_iqms', 'orig', 'hmc_epi', 'epi_mean', 'brainmask', 'hmc_fd', 'epi_parc',
        'in_dvars', 'in_stddev', 'outliers', 'in_spikes', 'in_fft',
        'exclude_index', 'mni_report']),
        name='inputnode')

    spmask = pe.Node(niu.Function(
        input_names=['in_file', 'in_mask'], output_names=['out_file', 'out_plot'],
        function=spikes_mask), name='SpikesMask')
    spikes_bg = pe.Node(Spikes(no_zscore=True, detrend=False), name='SpikesFinderBgMask')

    bigplot = pe.Node(niu.Function(
        input_names=['in_func', 'in_mask', 'in_segm', 'in_spikes_bg',
                     'fd', 'dvars', 'outliers'],
        output_names=['out_file'], function=_big_plot), name='BigPlot')

    workflow.connect([
        (inputnode, spikes_bg, [('orig', 'in_file')]),
        (inputnode, spmask, [('orig', 'in_file')]),
        (inputnode, bigplot, [('hmc_epi', 'in_func'),
                              ('brainmask', 'in_mask'),
                              ('hmc_fd', 'fd'),
                              ('in_dvars', 'dvars'),
                              ('epi_parc', 'in_segm'),
                              ('outliers', 'outliers')]),
        (spikes_bg, bigplot, [('out_tsz', 'in_spikes_bg')]),
        (spmask, spikes_bg, [('out_file', 'in_mask')]),
    ])


    mosaic_mean = pe.Node(PlotMosaic(
        out_file='plot_func_mean_mosaic1.svg',
        title='EPI mean session',
        cmap='Greys_r'),
        name='PlotMosaicMean')

    mosaic_stddev = pe.Node(PlotMosaic(
        out_file='plot_func_stddev_mosaic2_stddev.svg',
        title='EPI SD session',
        cmap='viridis'), name='PlotMosaicSD')

    mplots = pe.Node(niu.Merge(pages + extra_pages + int(settings.get('fft_spikes_detector', False))),
                     name='MergePlots')
    rnode = pe.Node(niu.Function(
        input_names=['in_iqms', 'in_plots', 'exclude_index', 'wf_details'],
        output_names=['out_file'], function=individual_html), name='GenerateReport')
    wf_details = []
    if settings.get('hmc_afni', False):
        wf_details.append('Framewise Displacement was computed using AFNI <code>3dvolreg</code>')
    else:
        wf_details.append('Framewise Displacement was computed using FSL <code>mcflirt</code>')

    rnode.inputs.wf_details = wf_details

    # Link images that should be reported
    dsplots = pe.Node(nio.DataSink(
        base_directory=settings['output_dir'], parameterization=False), name='dsplots')
    dsplots.inputs.container = 'reports'

    workflow.connect([
        (inputnode, rnode, [('in_iqms', 'in_iqms'),
                            ('exclude_index', 'exclude_index')]),
        (inputnode, mosaic_mean, [('epi_mean', 'in_file')]),
        (inputnode, mosaic_stddev, [('in_stddev', 'in_file')]),
        (mosaic_mean, mplots, [('out_file', 'in1')]),
        (mosaic_stddev, mplots, [('out_file', 'in2')]),
        (bigplot, mplots, [('out_file', 'in3')]),
        (mplots, rnode, [('out', 'in_plots')]),
        (rnode, dsplots, [('out_file', '@html_report')]),
    ])

    if settings.get('fft_spikes_detector', False):
        mosaic_spikes = pe.Node(PlotSpikes(
            out_file='plot_spikes.svg', cmap='viridis',
            title='High-Frequency spikes'),
            name='PlotSpikes')

        workflow.connect([
            (inputnode, mosaic_spikes, [('orig', 'in_file'),
                                        ('in_spikes', 'in_spikes'),
                                        ('in_fft', 'in_fft')]),
            (mosaic_spikes, mplots, [('out_file', 'in4')])
        ])

    if not verbose:
        return workflow

    mosaic_zoom = pe.Node(PlotMosaic(
        out_file='plot_anat_mosaic1_zoomed.svg',
        title='Zoomed-in EPI mean',
        cmap='Greys_r'), name='PlotMosaicZoomed')

    mosaic_noise = pe.Node(PlotMosaic(
        out_file='plot_anat_mosaic2_noise.svg',
        title='Enhanced noise in EPI mean',
        only_noise=True, cmap='viridis_r'), name='PlotMosaicNoise')

    # Verbose-reporting goes here
    from mriqc.interfaces.viz import PlotContours
    from mriqc.viz.utils import plot_bg_dist

    plot_bmask = pe.Node(PlotContours(
        display_mode='z', levels=[.5], colors=['r'], cut_coords=10,
        out_file='bmask'), name='PlotBrainmask')

    workflow.connect([
        (inputnode, plot_bmask, [('epi_mean', 'in_file'),
                                 ('brainmask', 'in_contours')]),
        (inputnode, mosaic_zoom, [('epi_mean', 'in_file'),
                                  ('brainmask', 'bbox_mask_file')]),
        (inputnode, mosaic_noise, [('epi_mean', 'in_file')]),
        (mosaic_zoom, mplots, [('out_file', 'in%d' % (pages + 1))]),
        (mosaic_noise, mplots, [('out_file', 'in%d' % (pages + 2))]),
        (plot_bmask, mplots, [('out_file', 'in%d' % (pages + 3))]),
        (inputnode, mplots, [('mni_report', 'in%d' % (pages + 4))]),
    ])
    return workflow


def fmri_bmsk_workflow(name='fMRIBrainMask', use_bet=False):
    """
    Computes a brain mask for the input :abbr:`fMRI (functional MRI)`
    dataset

    .. workflow::

      from mriqc.workflows.functional import fmri_bmsk_workflow
      wf = fmri_bmsk_workflow()


    """

    workflow = pe.Workflow(name=name)
    inputnode = pe.Node(niu.IdentityInterface(fields=['in_file']),
                        name='inputnode')
    outputnode = pe.Node(niu.IdentityInterface(fields=['out_file']),
                         name='outputnode')

    if not use_bet:
        afni_msk = pe.Node(afni.Automask(
            outputtype='NIFTI_GZ'), name='afni_msk')

        # Connect brain mask extraction
        workflow.connect([
            (inputnode, afni_msk, [('in_file', 'in_file')]),
            (afni_msk, outputnode, [('out_file', 'out_file')])
        ])

    else:
        from nipype.interfaces.fsl import BET, ErodeImage
        bet_msk = pe.Node(BET(mask=True, functional=True), name='bet_msk')
        erode = pe.Node(ErodeImage(kernel_shape='box', kernel_size=1.0),
                        name='erode')

        # Connect brain mask extraction
        workflow.connect([
            (inputnode, bet_msk, [('in_file', 'in_file')]),
            (bet_msk, erode, [('mask_file', 'in_file')]),
            (erode, outputnode, [('out_file', 'out_file')])
        ])

    return workflow


def hmc_mcflirt(name='fMRI_HMC_mcflirt'):
    """
    An :abbr:`HMC (head motion correction)` for functional scans
    using FSL MCFLIRT

    .. workflow::

      from mriqc.workflows.functional import hmc_mcflirt
      wf = hmc_mcflirt()

    """

    workflow = pe.Workflow(name=name)

    inputnode = pe.Node(niu.IdentityInterface(
        fields=['in_file', 'fd_radius', 'start_idx', 'stop_idx']), name='inputnode')

    outputnode = pe.Node(niu.IdentityInterface(
        fields=['out_file', 'out_fd']), name='outputnode')

    mcflirt = pe.Node(fsl.MCFLIRT(save_plots=True, save_rms=True, save_mats=True),
                      name='MCFLIRT')
    fdnode = pe.Node(nac.FramewiseDisplacement(normalize=False,
                                               parameter_source="FSL"),
                     name='ComputeFD')

    workflow.connect([
        (inputnode, mcflirt, [('in_file', 'in_file')]),
        (inputnode, fdnode, [('fd_radius', 'radius')]),
        (mcflirt, fdnode, [('par_file', 'in_file')]),
        (mcflirt, outputnode, [('out_file', 'out_file')]),
        (fdnode, outputnode, [('out_file', 'out_fd')]),
    ])

    return workflow


def hmc_afni(name='fMRI_HMC_afni', st_correct=False, despike=False,
             deoblique=False, start_idx=None, stop_idx=None):
    """
    A :abbr:`HMC (head motion correction)` workflow for
    functional scans

    .. workflow::

      from mriqc.workflows.functional import hmc_afni
      wf = hmc_afni()

    """

    workflow = pe.Workflow(name=name)

    inputnode = pe.Node(niu.IdentityInterface(
        fields=['in_file', 'fd_radius', 'start_idx', 'stop_idx']), name='inputnode')

    outputnode = pe.Node(niu.IdentityInterface(
        fields=['out_file', 'out_fd']), name='outputnode')

    if (start_idx is not None) or (stop_idx is not None):
        drop_trs = pe.Node(afni.Calc(expr='a', outputtype='NIFTI_GZ'),
                           name='drop_trs')
        workflow.connect([
            (inputnode, drop_trs, [('in_file', 'in_file_a'),
                                   ('start_idx', 'start_idx'),
                                   ('stop_idx', 'stop_idx')]),
        ])
    else:
        drop_trs = pe.Node(niu.IdentityInterface(fields=['out_file']),
                              name='drop_trs')
        workflow.connect([
            (inputnode, drop_trs, [('in_file', 'out_file')]),
        ])

    get_mean_RPI = pe.Node(afni.TStat(
        options='-mean', outputtype='NIFTI_GZ'), name='get_mean_RPI')

    # calculate hmc parameters
    hmc = pe.Node(
        afni.Volreg(args='-Fourier -twopass', zpad=4, outputtype='NIFTI_GZ'),
        name='motion_correct')

    get_mean_motion = get_mean_RPI.clone('get_mean_motion')
    hmc_A = hmc.clone('motion_correct_A')
    hmc_A.inputs.md1d_file = 'max_displacement.1D'

    # Compute the frame-wise displacement
    fdnode = pe.Node(nac.FramewiseDisplacement(normalize=False,
                                               parameter_source="AFNI"),
                     name='ComputeFD')

    workflow.connect([
        (inputnode, fdnode, [('fd_radius', 'radius')]),
        (get_mean_RPI, hmc, [('out_file', 'basefile')]),
        (hmc, get_mean_motion, [('out_file', 'in_file')]),
        (get_mean_motion, hmc_A, [('out_file', 'basefile')]),
        (hmc_A, outputnode, [('out_file', 'out_file')]),
        (hmc_A, fdnode, [('oned_file', 'in_file')]),
        (fdnode, outputnode, [('out_file', 'out_fd')]),
    ])

    # Slice timing correction, despiking, and deoblique

    st_corr = pe.Node(afni.TShift(outputtype='NIFTI_GZ'), name='TimeShifts')

    deoblique_node = pe.Node(afni.Refit(deoblique=True), name='deoblique')

    despike_node = pe.Node(afni.Despike(outputtype='NIFTI_GZ'), name='despike')

    if st_correct and despike and deoblique:

        workflow.connect([
            (drop_trs, st_corr, [('out_file', 'in_file')]),
            (st_corr, despike_node, [('out_file', 'in_file')]),
            (despike_node, deoblique_node, [('out_file', 'in_file')]),
            (deoblique_node, get_mean_RPI, [('out_file', 'in_file')]),
            (deoblique_node, hmc, [('out_file', 'in_file')]),
            (deoblique_node, hmc_A, [('out_file', 'in_file')]),
        ])

    elif st_correct and despike:

        workflow.connect([
            (drop_trs, st_corr, [('out_file', 'in_file')]),
            (st_corr, despike_node, [('out_file', 'in_file')]),
            (despike_node, get_mean_RPI, [('out_file', 'in_file')]),
            (despike_node, hmc, [('out_file', 'in_file')]),
            (despike_node, hmc_A, [('out_file', 'in_file')]),
        ])

    elif st_correct and deoblique:

        workflow.connect([
            (drop_trs, st_corr, [('out_file', 'in_file')]),
            (st_corr, deoblique_node, [('out_file', 'in_file')]),
            (deoblique_node, get_mean_RPI, [('out_file', 'in_file')]),
            (deoblique_node, hmc, [('out_file', 'in_file')]),
            (deoblique_node, hmc_A, [('out_file', 'in_file')]),
        ])

    elif st_correct:

        workflow.connect([
            (drop_trs, st_corr, [('out_file', 'in_file')]),
            (st_corr, get_mean_RPI, [('out_file', 'in_file')]),
            (st_corr, hmc, [('out_file', 'in_file')]),
            (st_corr, hmc_A, [('out_file', 'in_file')]),
        ])

    elif despike and deoblique:

        workflow.connect([
            (drop_trs, despike_node, [('out_file', 'in_file')]),
            (despike_node, deoblique_node, [('out_file', 'in_file')]),
            (deoblique_node, get_mean_RPI, [('out_file', 'in_file')]),
            (deoblique_node, hmc, [('out_file', 'in_file')]),
            (deoblique_node, hmc_A, [('out_file', 'in_file')]),
        ])

    elif despike:

        workflow.connect([
            (drop_trs, despike_node, [('out_file', 'in_file')]),
            (despike_node, get_mean_RPI, [('out_file', 'in_file')]),
            (despike_node, hmc, [('out_file', 'in_file')]),
            (despike_node, hmc_A, [('out_file', 'in_file')]),
        ])

    elif deoblique:

        workflow.connect([
            (drop_trs, deoblique_node, [('out_file', 'in_file')]),
            (deoblique_node, get_mean_RPI, [('out_file', 'in_file')]),
            (deoblique_node, hmc, [('out_file', 'in_file')]),
            (deoblique_node, hmc_A, [('out_file', 'in_file')]),

        ])

    else:

        workflow.connect([
            (drop_trs, get_mean_RPI, [('out_file', 'in_file')]),
            (drop_trs, hmc, [('out_file', 'in_file')]),
            (drop_trs, hmc_A, [('out_file', 'in_file')]),
        ])

    return workflow

def epi_mni_align(name='SpatialNormalization', ants_nthreads=6, testing=False, resolution=2):
    """
    Uses FSL FLIRT with the BBR cost function to find the transform that
    maps the EPI space into the MNI152-nonlinear-symmetric atlas.

    The input epi_mean is the averaged and brain-masked EPI timeseries

    Returns the EPI mean resampled in MNI space (for checking out registration) and
    the associated "lobe" parcellation in EPI space.

    .. workflow::

      from mriqc.workflows.functional import epi_mni_align
      wf = epi_mni_align()

    """
    from nipype.interfaces.ants import ApplyTransforms, N4BiasFieldCorrection
    from niworkflows.data import get_mni_icbm152_nlin_asym_09c as get_template
    from niworkflows.interfaces.registration import RobustMNINormalizationRPT as RobustMNINormalization
    from pkg_resources import resource_filename as pkgrf

    mni_template = get_template()

    workflow = pe.Workflow(name=name)
    inputnode = pe.Node(niu.IdentityInterface(fields=['epi_mean', 'epi_mask']),
                        name='inputnode')
    outputnode = pe.Node(niu.IdentityInterface(
        fields=['epi_mni', 'epi_parc', 'report']), name='outputnode')

    epimask = pe.Node(fsl.ApplyMask(), name='EPIApplyMask')

    n4itk = pe.Node(N4BiasFieldCorrection(dimension=3), name='SharpenEPI')

    norm = pe.Node(RobustMNINormalization(
        num_threads=ants_nthreads, template='mni_icbm152_nlin_asym_09c',
        testing=testing, moving='EPI', generate_report=True),
                   name='EPI2MNI')
    norm.inputs.reference_image = pkgrf(
        'mriqc', 'data/mni/%dmm_T2_brain.nii.gz' % resolution)

    # Warp segmentation into EPI space
    invt = pe.Node(ApplyTransforms(
        input_image=op.join(mni_template, '%dmm_parc.nii.gz' % resolution),
        dimension=3, default_value=0, interpolation='NearestNeighbor'),
                   name='ResampleSegmentation')

    workflow.connect([
        (inputnode, invt, [('epi_mean', 'reference_image')]),
        (inputnode, n4itk, [('epi_mean', 'input_image')]),
        (inputnode, epimask, [('epi_mask', 'mask_file')]),
        (n4itk, epimask, [('output_image', 'in_file')]),
        (epimask, norm, [('out_file', 'moving_image')]),
        (norm, invt, [
            ('inverse_composite_transform', 'transforms')]),
        (invt, outputnode, [('output_image', 'epi_parc')]),
        (norm, outputnode, [('warped_image', 'epi_mni'),
                            ('out_report', 'report')]),

    ])
    return workflow


def spikes_mask(in_file, in_mask=None, out_file=None):
    """
    Utility function to calculate a mask in which check
    for :abbr:`EM (electromagnetic)` spikes.
    """

    import os.path as op
    import nibabel as nb
    import numpy as np
    from nilearn.image import mean_img
    from nilearn.plotting import plot_roi
    from scipy import ndimage as nd

    if out_file is None:
        fname, ext = op.splitext(op.basename(in_file))
        if ext == '.gz':
            fname, ext2 = op.splitext(fname)
            ext = ext2 + ext
        out_file = op.abspath('{}_spmask{}'.format(fname, ext))
        out_plot = op.abspath('{}_spmask.pdf'.format(fname))

    in_4d_nii = nb.load(in_file)
    orientation = nb.aff2axcodes(in_4d_nii.affine)

    if in_mask:
        mask_data = nb.load(in_mask).get_data()
        a = np.where(mask_data != 0)
        bbox = np.max(a[0]) - np.min(a[0]), np.max(a[1]) - \
            np.min(a[1]), np.max(a[2]) - np.min(a[2])
        longest_axis = np.argmax(bbox)

        # Input here is a binarized and intersected mask data from previous section
        dil_mask = nd.binary_dilation(
            mask_data, iterations=int(mask_data.shape[longest_axis]/9))

        rep = list(mask_data.shape)
        rep[longest_axis] = -1
        new_mask_2d = dil_mask.max(axis=longest_axis).reshape(rep)

        rep = [1, 1, 1]
        rep[longest_axis] = mask_data.shape[longest_axis]
        new_mask_3d = np.logical_not(np.tile(new_mask_2d, rep))
    else:
        new_mask_3d = np.zeros(in_4d_nii.shape[:3]) == 1

    if orientation[0] in ['L', 'R']:
        new_mask_3d[0:2, :, :] = True
        new_mask_3d[-3:-1, :, :] = True
    else:
        new_mask_3d[:, 0:2, :] = True
        new_mask_3d[:, -3:-1, :] = True

    mask_nii = nb.Nifti1Image(new_mask_3d.astype(np.uint8), in_4d_nii.get_affine(),
                              in_4d_nii.get_header())
    mask_nii.to_filename(out_file)

    plot_roi(mask_nii, mean_img(in_4d_nii), output_file=out_plot)
    return out_file, out_plot


def _mean(inlist):
    import numpy as np
    return np.mean(inlist)


def _parse_tqual(in_file):
    import numpy as np
    with open(in_file, 'r') as fin:
        lines = fin.readlines()
        # remove general information
        lines = [l for l in lines if l[:2] != '++']
        # remove general information and warnings
        return np.mean([float(l.strip()) for l in lines])
    raise RuntimeError('AFNI 3dTqual was not parsed correctly')


def _parse_tout(in_file):
    import numpy as np
    data = np.loadtxt(in_file)  # pylint: disable=no-member
    return data.mean()


def _big_plot(in_func, in_mask, in_segm, in_spikes_bg,
              fd, dvars, outliers, out_file=None):
    import os.path as op
    import numpy as np
    from mriqc.viz.fmriplots import fMRIPlot
    if out_file is None:
        fname, ext = op.splitext(op.basename(in_func))
        if ext == '.gz':
            fname, _ = op.splitext(fname)
        out_file = op.abspath('{}_fmriplot.svg'.format(fname))

    title = 'fMRI Summary plot'

    myplot = fMRIPlot(
        in_func, in_mask, in_segm, title=title)
    myplot.add_spikes(np.loadtxt(in_spikes_bg), zscored=False)

    # Add AFNI ouliers plot
    myplot.add_confounds([np.nan] + np.loadtxt(outliers, usecols=[0]).tolist(),
                         {'name': 'ouliers', 'units': '%', 'normalize': False,
                          'ylims': (0.0, None)})

    # Pick non-standardize dvars
    myplot.add_confounds([np.nan] + np.loadtxt(dvars, skiprows=1,
                                               usecols=[1]).tolist(),
                         {'name': 'DVARS', 'units': None, 'normalize': False})

    # Add FD
    myplot.add_confounds([np.nan] + np.loadtxt(fd, skiprows=1,
                                               usecols=[0]).tolist(),
                         {'name': 'FD', 'units': 'mm', 'normalize': False,
                          'cutoff': [0.2], 'ylims': (0.0, 0.2)})
    myplot.plot()
    myplot.fig.savefig(out_file, bbox_inches='tight')
    myplot.fig.clf()
    return out_file
