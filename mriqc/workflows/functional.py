# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
=======================
The functional workflow
=======================

.. image :: _static/functional_workflow_source.svg

The functional workflow follows the following steps:

#. Sanitize (revise data types and xforms) input data, read
   associated metadata and discard non-steady state frames.
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
from .. import config
from nipype.pipeline import engine as pe
from nipype.algorithms import confounds as nac
from nipype.interfaces import io as nio
from nipype.interfaces import utility as niu
from nipype.interfaces import afni, ants, fsl

from .utils import get_fwhmx
from ..interfaces import FunctionalQC, Spikes, IQMFileSink
from ..interfaces.reports import AddProvenance


def fmri_qc_workflow(name='funcMRIQC'):
    """
    The fMRI qc workflow

    .. workflow::

        import os.path as op
        from mriqc.workflows.functional import fmri_qc_workflow
        from mriqc.testing import mock_config
        with mock_config():
            wf = fmri_qc_workflow()


    """
    from niworkflows.interfaces.utils import SanitizeImage

    workflow = pe.Workflow(name=name)

    mem_gb = config.workflow.biggest_file_gb

    dataset = config.workflow.inputs.get("bold", [])
    config.loggers.workflow.info(f"""\
Building functional MRIQC workflow for files: {', '.join(dataset)}.""")

    # Define workflow, inputs and outputs
    # 0. Get data, put it in RAS orientation
    inputnode = pe.Node(niu.IdentityInterface(fields=['in_file']), name='inputnode')
    inputnode.iterables = [('in_file', dataset)]

    outputnode = pe.Node(niu.IdentityInterface(
        fields=['qc', 'mosaic', 'out_group', 'out_dvars',
                'out_fd']), name='outputnode')

    non_steady_state_detector = pe.Node(nac.NonSteadyStateDetector(),
                                        name="non_steady_state_detector")

    sanitize = pe.Node(SanitizeImage(), name="sanitize",
                       mem_gb=mem_gb * 4.0)
    sanitize.inputs.max_32bit = config.execution.float32

    # Workflow --------------------------------------------------------

    # 1. HMC: head motion correct
    if config.workflow.hmc_fsl:
        hmcwf = hmc_mcflirt()
    else:
        hmcwf = hmc_afni()

    # Set HMC settings
    hmcwf.inputs.inputnode.fd_radius = config.workflow.fd_radius

    mean = pe.Node(afni.TStat(                   # 2. Compute mean fmri
        options='-mean', outputtype='NIFTI_GZ'), name='mean',
        mem_gb=mem_gb * 1.5)
    skullstrip_epi = fmri_bmsk_workflow(use_bet=True)

    # EPI to MNI registration
    ema = epi_mni_align()

    # Compute TSNR using nipype implementation
    tsnr = pe.Node(nac.TSNR(), name='compute_tsnr', mem_gb=mem_gb * 2.5)

    # 7. Compute IQMs
    iqmswf = compute_iqms()
    # Reports
    repwf = individual_reports()

    workflow.connect([
        (inputnode, iqmswf, [('in_file', 'inputnode.in_file')]),
        (inputnode, sanitize, [('in_file', 'in_file')]),
        (inputnode, non_steady_state_detector, [('in_file', 'in_file')]),
        (non_steady_state_detector, sanitize, [('n_volumes_to_discard', 'n_volumes_to_discard')]),
        (sanitize, hmcwf, [('out_file', 'inputnode.in_file')]),
        (mean, skullstrip_epi, [('out_file', 'inputnode.in_file')]),
        (hmcwf, mean, [('outputnode.out_file', 'in_file')]),
        (hmcwf, tsnr, [('outputnode.out_file', 'in_file')]),
        (mean, ema, [('out_file', 'inputnode.epi_mean')]),
        (skullstrip_epi, ema, [('outputnode.out_file', 'inputnode.epi_mask')]),
        (sanitize, iqmswf, [('out_file', 'inputnode.in_ras')]),
        (mean, iqmswf, [('out_file', 'inputnode.epi_mean')]),
        (hmcwf, iqmswf, [('outputnode.out_file', 'inputnode.hmc_epi'),
                         ('outputnode.out_fd', 'inputnode.hmc_fd')]),
        (skullstrip_epi, iqmswf, [('outputnode.out_file', 'inputnode.brainmask')]),
        (tsnr, iqmswf, [('tsnr_file', 'inputnode.in_tsnr')]),
        (sanitize, repwf, [('out_file', 'inputnode.in_ras')]),
        (mean, repwf, [('out_file', 'inputnode.epi_mean')]),
        (tsnr, repwf, [('stddev_file', 'inputnode.in_stddev')]),
        (skullstrip_epi, repwf, [('outputnode.out_file', 'inputnode.brainmask')]),
        (hmcwf, repwf, [('outputnode.out_fd', 'inputnode.hmc_fd'),
                        ('outputnode.out_file', 'inputnode.hmc_epi')]),
        (ema, repwf, [('outputnode.epi_parc', 'inputnode.epi_parc'),
                      ('outputnode.report', 'inputnode.mni_report')]),
        (non_steady_state_detector, iqmswf, [('n_volumes_to_discard', 'inputnode.exclude_index')]),
        (iqmswf, repwf, [('outputnode.out_file', 'inputnode.in_iqms'),
                         ('outputnode.out_dvars', 'inputnode.in_dvars'),
                         ('outputnode.outliers', 'inputnode.outliers')]),
        (hmcwf, outputnode, [('outputnode.out_fd', 'out_fd')]),
    ])

    if config.workflow.fft_spikes_detector:
        workflow.connect([
            (iqmswf, repwf, [('outputnode.out_spikes', 'inputnode.in_spikes'),
                             ('outputnode.out_fft', 'inputnode.in_fft')]),
        ])

    if config.workflow.ica:
        from niworkflows.interfaces import segmentation as nws
        melodic = pe.Node(nws.MELODICRPT(no_bet=True,
                                         no_mask=True,
                                         no_mm=True,
                                         compress_report=False,
                                         generate_report=True),
                          name="ICA", mem_gb=max(mem_gb * 5, 8))
        workflow.connect([
            (sanitize, melodic, [('out_file', 'in_files')]),
            (skullstrip_epi, melodic, [('outputnode.out_file', 'report_mask')]),
            (melodic, repwf, [('out_report', 'inputnode.ica_report')])
        ])

    # Upload metrics
    if not config.execution.no_sub:
        from ..interfaces.webapi import UploadIQMs
        upldwf = pe.Node(UploadIQMs(), name='UploadMetrics')
        upldwf.inputs.url = config.execution.webapi_url
        upldwf.inputs.strict = config.execution.upload_strict
        if config.execution.webapi_port:
            upldwf.inputs.port = config.execution.webapi_port

        workflow.connect([
            (iqmswf, upldwf, [('outputnode.out_file', 'in_iqms')]),
        ])

    return workflow


def compute_iqms(name='ComputeIQMs'):
    """
    Workflow that actually computes the IQMs

    .. workflow::

        from mriqc.workflows.functional import compute_iqms
        from mriqc.testing import mock_config
        with mock_config():
            wf = compute_iqms()


    """
    from niworkflows.interfaces.bids import ReadSidecarJSON

    from .utils import _tofloat
    from ..interfaces.transitional import GCOR

    mem_gb = config.workflow.biggest_file_gb

    workflow = pe.Workflow(name=name)
    inputnode = pe.Node(niu.IdentityInterface(fields=[
        'in_file', 'in_ras',
        'epi_mean', 'brainmask', 'hmc_epi', 'hmc_fd', 'fd_thres', 'in_tsnr', 'metadata',
        'exclude_index']), name='inputnode')
    outputnode = pe.Node(niu.IdentityInterface(
        fields=['out_file', 'out_dvars', 'outliers', 'out_spikes', 'out_fft']),
        name='outputnode')

    # Set FD threshold
    inputnode.inputs.fd_thres = config.workflow.fd_thres

    # Compute DVARS
    dvnode = pe.Node(nac.ComputeDVARS(save_plot=False, save_all=True), name='ComputeDVARS',
                     mem_gb=mem_gb * 3)

    # AFNI quality measures
    fwhm_interface = get_fwhmx()
    fwhm = pe.Node(fwhm_interface, name='smoothness')
    # fwhm.inputs.acf = True  # add when AFNI >= 16
    outliers = pe.Node(afni.OutlierCount(fraction=True, out_file='outliers.out'),
                       name='outliers', mem_gb=mem_gb * 2.5)

    quality = pe.Node(afni.QualityIndex(automask=True), out_file='quality.out',
                      name='quality', mem_gb=mem_gb * 3)

    gcor = pe.Node(GCOR(), name='gcor', mem_gb=mem_gb * 2)

    measures = pe.Node(FunctionalQC(), name='measures', mem_gb=mem_gb * 3)

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
        (outliers, outputnode, [('out_file', 'outliers')])
    ])

    # Add metadata
    meta = pe.Node(ReadSidecarJSON(), name='metadata',
                   run_without_submitting=True)
    addprov = pe.Node(AddProvenance(modality="bold"),
                      name='provenance',
                      run_without_submitting=True)

    # Save to JSON file
    datasink = pe.Node(IQMFileSink(
        modality='bold', out_dir=str(config.execution.output_dir),
        dataset=config.execution.dsname),
        name='datasink', run_without_submitting=True)

    workflow.connect([
        (inputnode, datasink, [('in_file', 'in_file'),
                               ('exclude_index', 'dummy_trs')]),
        (inputnode, meta, [('in_file', 'in_file')]),
        (inputnode, addprov, [('in_file', 'in_file')]),
        (meta, datasink, [('subject', 'subject_id'),
                          ('session', 'session_id'),
                          ('task', 'task_id'),
                          ('acquisition', 'acq_id'),
                          ('reconstruction', 'rec_id'),
                          ('run', 'run_id'),
                          ('out_dict', 'metadata')]),
        (addprov, datasink, [('out_prov', 'provenance')]),
        (outliers, datasink, [(('out_file', _parse_tout), 'aor')]),
        (gcor, datasink, [(('out', _tofloat), 'gcor')]),
        (quality, datasink, [(('out_file', _parse_tqual), 'aqi')]),
        (measures, datasink, [('out_qc', 'root')]),
        (datasink, outputnode, [('out_file', 'out_file')])
    ])

    # FFT spikes finder
    if config.workflow.fft_spikes_detector:
        from .utils import slice_wise_fft
        spikes_fft = pe.Node(niu.Function(
            input_names=['in_file'],
            output_names=['n_spikes', 'out_spikes', 'out_fft'],
            function=slice_wise_fft), name='SpikesFinderFFT')

        workflow.connect([
            (inputnode, spikes_fft, [('in_ras', 'in_file')]),
            (spikes_fft, outputnode, [('out_spikes', 'out_spikes'),
                                      ('out_fft', 'out_fft')]),
            (spikes_fft, datasink, [('n_spikes', 'spikes_num')])
        ])
    return workflow


def individual_reports(name='ReportsWorkflow'):
    """
    Encapsulates nodes writing plots

    .. workflow::

        from mriqc.workflows.functional import individual_reports
        from mriqc.testing import mock_config
        with mock_config():
            wf = individual_reports()

    """
    from niworkflows.interfaces.plotting import FMRISummary
    from ..interfaces import PlotMosaic, PlotSpikes
    from ..interfaces.reports import IndividualReport

    verbose = config.execution.verbose_reports
    mem_gb = config.workflow.biggest_file_gb

    pages = 5
    extra_pages = int(verbose) * 4

    workflow = pe.Workflow(name=name)
    inputnode = pe.Node(niu.IdentityInterface(fields=[
        'in_iqms', 'in_ras', 'hmc_epi', 'epi_mean', 'brainmask', 'hmc_fd', 'fd_thres', 'epi_parc',
        'in_dvars', 'in_stddev', 'outliers', 'in_spikes', 'in_fft',
        'mni_report', 'ica_report']),
        name='inputnode')

    # Set FD threshold
    inputnode.inputs.fd_thres = config.workflow.fd_thres

    spmask = pe.Node(niu.Function(
        input_names=['in_file', 'in_mask'], output_names=['out_file', 'out_plot'],
        function=spikes_mask), name='SpikesMask', mem_gb=mem_gb * 3.5)

    spikes_bg = pe.Node(Spikes(no_zscore=True, detrend=False), name='SpikesFinderBgMask',
                        mem_gb=mem_gb * 2.5)

    bigplot = pe.Node(FMRISummary(), name='BigPlot', mem_gb=mem_gb * 3.5)
    workflow.connect([
        (inputnode, spikes_bg, [('in_ras', 'in_file')]),
        (inputnode, spmask, [('in_ras', 'in_file')]),
        (inputnode, bigplot, [('hmc_epi', 'in_func'),
                              ('brainmask', 'in_mask'),
                              ('hmc_fd', 'fd'),
                              ('fd_thres', 'fd_thres'),
                              ('in_dvars', 'dvars'),
                              ('epi_parc', 'in_segm'),
                              ('outliers', 'outliers')]),
        (spikes_bg, bigplot, [('out_tsz', 'in_spikes_bg')]),
        (spmask, spikes_bg, [('out_file', 'in_mask')]),
    ])

    mosaic_mean = pe.Node(PlotMosaic(
        out_file='plot_func_mean_mosaic1.svg',
        cmap='Greys_r'),
        name='PlotMosaicMean')

    mosaic_stddev = pe.Node(PlotMosaic(
        out_file='plot_func_stddev_mosaic2_stddev.svg',
        cmap='viridis'), name='PlotMosaicSD')

    mplots = pe.Node(niu.Merge(pages + extra_pages + int(
        config.workflow.fft_spikes_detector) + int(
        config.workflow.ica)), name='MergePlots')
    rnode = pe.Node(IndividualReport(), name='GenerateReport')

    # Link images that should be reported
    dsplots = pe.Node(nio.DataSink(
        base_directory=str(config.execution.output_dir),
        parameterization=False),
        name='dsplots', run_without_submitting=True)

    workflow.connect([
        (inputnode, rnode, [('in_iqms', 'in_iqms')]),
        (inputnode, mosaic_mean, [('epi_mean', 'in_file')]),
        (inputnode, mosaic_stddev, [('in_stddev', 'in_file')]),
        (mosaic_mean, mplots, [('out_file', 'in1')]),
        (mosaic_stddev, mplots, [('out_file', 'in2')]),
        (bigplot, mplots, [('out_file', 'in3')]),
        (mplots, rnode, [('out', 'in_plots')]),
        (rnode, dsplots, [('out_file', '@html_report')]),
    ])

    if config.workflow.fft_spikes_detector:
        mosaic_spikes = pe.Node(PlotSpikes(
            out_file='plot_spikes.svg', cmap='viridis',
            title='High-Frequency spikes'),
            name='PlotSpikes')

        workflow.connect([
            (inputnode, mosaic_spikes, [('in_ras', 'in_file'),
                                        ('in_spikes', 'in_spikes'),
                                        ('in_fft', 'in_fft')]),
            (mosaic_spikes, mplots, [('out_file', 'in4')])
        ])

    if config.workflow.ica:
        page_number = 4 + config.workflow.fft_spikes_detector
        workflow.connect([
            (inputnode, mplots, [('ica_report', 'in%d' % page_number)])
        ])

    if not verbose:
        return workflow

    mosaic_zoom = pe.Node(PlotMosaic(
        out_file='plot_anat_mosaic1_zoomed.svg',
        cmap='Greys_r'), name='PlotMosaicZoomed')

    mosaic_noise = pe.Node(PlotMosaic(
        out_file='plot_anat_mosaic2_noise.svg',
        only_noise=True, cmap='viridis_r'), name='PlotMosaicNoise')

    # Verbose-reporting goes here
    from ..interfaces.viz import PlotContours

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
        from mriqc.testing import mock_config
        with mock_config():
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
        bet_msk = pe.Node(fsl.BET(mask=True, functional=True), name='bet_msk')
        erode = pe.Node(fsl.ErodeImage(), name='erode')

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
        from mriqc.testing import mock_config
        with mock_config():
            wf = hmc_mcflirt()

    """
    from niworkflows.interfaces.registration import EstimateReferenceImage

    workflow = pe.Workflow(name=name)

    inputnode = pe.Node(niu.IdentityInterface(
        fields=['in_file', 'fd_radius', 'start_idx', 'stop_idx']), name='inputnode')

    outputnode = pe.Node(niu.IdentityInterface(
        fields=['out_file', 'out_fd']), name='outputnode')

    gen_ref = pe.Node(EstimateReferenceImage(mc_method="AFNI"), name="gen_ref")

    mcflirt = pe.Node(fsl.MCFLIRT(save_plots=True, interpolation='sinc'),
                      name='MCFLIRT',
                      mem_gb=config.workflow.biggest_file_gb * 2.5)

    fdnode = pe.Node(nac.FramewiseDisplacement(normalize=False,
                                               parameter_source="FSL"),
                     name='ComputeFD')

    workflow.connect([
        (inputnode, gen_ref, [('in_file', 'in_file')]),
        (gen_ref, mcflirt, [('ref_image', 'ref_file')]),
        (inputnode, mcflirt, [('in_file', 'in_file')]),
        (inputnode, fdnode, [('fd_radius', 'radius')]),
        (mcflirt, fdnode, [('par_file', 'in_file')]),
        (mcflirt, outputnode, [('out_file', 'out_file')]),
        (fdnode, outputnode, [('out_file', 'out_fd')]),
    ])

    return workflow


def hmc_afni(name='fMRI_HMC_afni'):
    """
    A :abbr:`HMC (head motion correction)` workflow for
    functional scans

    .. workflow::

        from mriqc.workflows.functional import hmc_afni
        from mriqc.testing import mock_config
        with mock_config():
            wf = hmc_afni()

    """
    from niworkflows.interfaces.registration import EstimateReferenceImage

    mem_gb = config.workflow.biggest_file_gb

    workflow = pe.Workflow(name=name)

    inputnode = pe.Node(niu.IdentityInterface(
        fields=['in_file', 'fd_radius', 'start_idx', 'stop_idx']), name='inputnode')

    outputnode = pe.Node(niu.IdentityInterface(
        fields=['out_file', 'out_fd']), name='outputnode')

    if any((
        config.workflow.start_idx is not None,
        config.workflow.stop_idx is not None
    )):
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

    gen_ref = pe.Node(EstimateReferenceImage(mc_method="AFNI"), name="gen_ref")

    # calculate hmc parameters
    hmc = pe.Node(
        afni.Volreg(args='-Fourier -twopass', zpad=4, outputtype='NIFTI_GZ'),
        name='motion_correct', mem_gb=mem_gb * 2.5)

    # Compute the frame-wise displacement
    fdnode = pe.Node(nac.FramewiseDisplacement(normalize=False,
                                               parameter_source="AFNI"),
                     name='ComputeFD')

    workflow.connect([
        (inputnode, fdnode, [('fd_radius', 'radius')]),
        (gen_ref, hmc, [('ref_image', 'basefile')]),
        (hmc, outputnode, [('out_file', 'out_file')]),
        (hmc, fdnode, [('oned_file', 'in_file')]),
        (fdnode, outputnode, [('out_file', 'out_fd')]),
    ])

    # Slice timing correction, despiking, and deoblique

    st_corr = pe.Node(afni.TShift(outputtype='NIFTI_GZ'), name='TimeShifts')

    deoblique_node = pe.Node(afni.Refit(deoblique=True), name='deoblique')

    despike_node = pe.Node(afni.Despike(outputtype='NIFTI_GZ'), name='despike')

    if all((
        config.workflow.correct_slice_timing,
        config.workflow.despike,
        config.workflow.deoblique
    )):

        workflow.connect([
            (drop_trs, st_corr, [('out_file', 'in_file')]),
            (st_corr, despike_node, [('out_file', 'in_file')]),
            (despike_node, deoblique_node, [('out_file', 'in_file')]),
            (deoblique_node, gen_ref, [('out_file', 'in_file')]),
            (deoblique_node, hmc, [('out_file', 'in_file')]),
        ])
    elif config.workflow.correct_slice_timing and config.workflow.despike:

        workflow.connect([
            (drop_trs, st_corr, [('out_file', 'in_file')]),
            (st_corr, despike_node, [('out_file', 'in_file')]),
            (despike_node, gen_ref, [('out_file', 'in_file')]),
            (despike_node, hmc, [('out_file', 'in_file')]),
        ])

    elif config.workflow.correct_slice_timing and config.workflow.deoblique:

        workflow.connect([
            (drop_trs, st_corr, [('out_file', 'in_file')]),
            (st_corr, deoblique_node, [('out_file', 'in_file')]),
            (deoblique_node, gen_ref, [('out_file', 'in_file')]),
            (deoblique_node, hmc, [('out_file', 'in_file')]),
        ])

    elif config.workflow.correct_slice_timing:

        workflow.connect([
            (drop_trs, st_corr, [('out_file', 'in_file')]),
            (st_corr, gen_ref, [('out_file', 'in_file')]),
            (st_corr, hmc, [('out_file', 'in_file')]),
        ])

    elif config.workflow.despike and config.workflow.deoblique:

        workflow.connect([
            (drop_trs, despike_node, [('out_file', 'in_file')]),
            (despike_node, deoblique_node, [('out_file', 'in_file')]),
            (deoblique_node, gen_ref, [('out_file', 'in_file')]),
            (deoblique_node, hmc, [('out_file', 'in_file')]),
        ])

    elif config.workflow.despike:

        workflow.connect([
            (drop_trs, despike_node, [('out_file', 'in_file')]),
            (despike_node, gen_ref, [('out_file', 'in_file')]),
            (despike_node, hmc, [('out_file', 'in_file')]),
        ])

    elif config.workflow.deoblique:

        workflow.connect([
            (drop_trs, deoblique_node, [('out_file', 'in_file')]),
            (deoblique_node, gen_ref, [('out_file', 'in_file')]),
            (deoblique_node, hmc, [('out_file', 'in_file')]),
        ])

    else:
        workflow.connect([
            (drop_trs, gen_ref, [('out_file', 'in_file')]),
            (drop_trs, hmc, [('out_file', 'in_file')]),
        ])

    return workflow


def epi_mni_align(name='SpatialNormalization'):
    """
    Uses FSL FLIRT with the BBR cost function to find the transform that
    maps the EPI space into the MNI152-nonlinear-symmetric atlas.

    The input epi_mean is the averaged and brain-masked EPI timeseries

    Returns the EPI mean resampled in MNI space (for checking out registration) and
    the associated "lobe" parcellation in EPI space.

    .. workflow::

        from mriqc.workflows.functional import epi_mni_align
        from mriqc.testing import mock_config
        with mock_config():
            wf = epi_mni_align()

    """
    from templateflow.api import get as get_template
    from niworkflows.interfaces.registration import (
        RobustMNINormalizationRPT as RobustMNINormalization
    )

    # Get settings
    testing = config.execution.debug
    n_procs = config.nipype.nprocs
    ants_nthreads = config.nipype.omp_nthreads

    workflow = pe.Workflow(name=name)
    inputnode = pe.Node(niu.IdentityInterface(fields=['epi_mean', 'epi_mask']),
                        name='inputnode')
    outputnode = pe.Node(niu.IdentityInterface(
        fields=['epi_mni', 'epi_parc', 'report']), name='outputnode')

    n4itk = pe.Node(ants.N4BiasFieldCorrection(dimension=3, copy_header=True),
                    name='SharpenEPI')

    norm = pe.Node(RobustMNINormalization(
        explicit_masking=False,
        flavor='testing' if testing else 'precise',
        float=config.execution.ants_float,
        generate_report=True,
        moving='boldref',
        num_threads=ants_nthreads,
        reference='boldref',
        reference_image=str(get_template(
            'MNI152NLin2009cAsym', resolution=2, suffix='boldref')),
        reference_mask=str(get_template(
            'MNI152NLin2009cAsym', resolution=2, desc='brain', suffix='mask')),
        template='MNI152NLin2009cAsym',
        template_resolution=2, ),
        name='EPI2MNI', num_threads=n_procs, mem_gb=3)

    # Warp segmentation into EPI space
    invt = pe.Node(ants.ApplyTransforms(
        float=True,
        input_image=str(get_template('MNI152NLin2009cAsym', resolution=1,
                                     desc='carpet', suffix='dseg')),
        dimension=3, default_value=0, interpolation='MultiLabel'),
        name='ResampleSegmentation')

    workflow.connect([
        (inputnode, invt, [('epi_mean', 'reference_image')]),
        (inputnode, n4itk, [('epi_mean', 'input_image')]),
        (inputnode, norm, [('epi_mask', 'moving_mask')]),
        (n4itk, norm, [('output_image', 'moving_image')]),
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
            mask_data, iterations=int(mask_data.shape[longest_axis] / 9))

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

    mask_nii = nb.Nifti1Image(new_mask_3d.astype(np.uint8), in_4d_nii.affine,
                              in_4d_nii.header)
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
