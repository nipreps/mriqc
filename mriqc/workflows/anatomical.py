#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# @Author: oesteban
# @Date:   2016-01-05 11:24:05
# @Email:  code@oscaresteban.es
# @Last modified by:   oesteban
"""
=======================
The anatomical workflow
=======================

.. image :: _static/anatomical_workflow_source.svg

The anatomical workflow follows the following steps:

#. Conform (reorientations, revise data types) input data and read
   associated metadata.
#. Skull-stripping (AFNI).
#. Calculate head mask -- :py:func:`headmsk_wf`.
#. Spatial Normalization to MNI (ANTs)
#. Calculate air mask above the nasial-cerebelum plane -- :py:func:`airmsk_wf`.
#. Brain tissue segmentation (FAST).
#. Extraction of IQMs -- :py:func:`compute_iqms`.
#. Individual-reports generation -- :py:func:`individual_reports`.

This workflow is orchestrated by :py:func:`anat_qc_workflow`.

For the skull-stripping, we use ``afni_wf`` from ``niworkflows.anat.skullstrip``:

.. workflow::

    import os.path as op
    from niworkflows.anat.skullstrip import afni_wf
    wf = afni_wf()


"""
from __future__ import print_function, division, absolute_import, unicode_literals
from builtins import zip, range
import os.path as op

from niworkflows.nipype.pipeline import engine as pe
from niworkflows.nipype.interfaces import io as nio
from niworkflows.nipype.interfaces import utility as niu
from niworkflows.nipype.interfaces import fsl, ants, afni
from niworkflows.data import get_mni_icbm152_nlin_asym_09c
from niworkflows.anat.skullstrip import afni_wf as skullstrip_wf
from niworkflows.interfaces.registration import RobustMNINormalizationRPT as RobustMNINormalization

from .. import DEFAULTS, logging
from ..interfaces import (StructuralQC, ArtifactMask, ReadSidecarJSON,
                          ConformImage, ComputeQI2, IQMFileSink, RotationMask)
from ..utils.misc import check_folder
WFLOGGER = logging.getLogger('mriqc.workflow')


def anat_qc_workflow(dataset, settings, mod='T1w', name='anatMRIQC'):
    """
    One-subject-one-session-one-run pipeline to extract the NR-IQMs from
    anatomical images

    .. workflow::

        import os.path as op
        from mriqc.workflows.anatomical import anat_qc_workflow
        datadir = op.abspath('data')
        wf = anat_qc_workflow([op.join(datadir, 'sub-001/anat/sub-001_T1w.nii.gz')],
                              settings={'bids_dir': datadir,
                                        'output_dir': op.abspath('out'),
                                        'ants_nthreads': 1,
                                        'no_sub': True})

    """

    workflow = pe.Workflow(name=name+mod)
    WFLOGGER.info('Building anatomical MRI QC workflow, datasets list: %s',
                  sorted([d.replace(settings['bids_dir'] + '/', '') for d in dataset]))

    # Define workflow, inputs and outputs
    # 0. Get data
    inputnode = pe.Node(niu.IdentityInterface(fields=['in_file']), name='inputnode')
    inputnode.iterables = [('in_file', dataset)]

    outputnode = pe.Node(niu.IdentityInterface(fields=['out_json']), name='outputnode')

    # 1. Reorient anatomical image
    to_ras = pe.Node(ConformImage(check_dtype=False), name='conform')
    # 2. Skull-stripping (afni)
    asw = skullstrip_wf(n4_nthreads=settings.get('ants_nthreads', 1), unifize=False)
    # 3. Head mask
    hmsk = headmsk_wf()
    # 4. Spatial Normalization, using ANTs
    norm = spatial_normalization(settings)
    # 5. Air mask (with and without artifacts)
    amw = airmsk_wf()
    # 6. Brain tissue segmentation
    segment = pe.Node(fsl.FAST(segments=True, out_basename='segment', img_type=int(mod[1])),
                      name='segmentation', estimated_memory_gb=3)
    # 7. Compute IQMs
    iqmswf = compute_iqms(settings, modality=mod)
    # Reports
    repwf = individual_reports(settings)

    # Connect all nodes
    workflow.connect([
        (inputnode, to_ras, [('in_file', 'in_file')]),
        (inputnode, iqmswf, [('in_file', 'inputnode.in_file')]),
        (to_ras, asw, [('out_file', 'inputnode.in_file')]),
        (asw, segment, [('outputnode.out_file', 'in_files')]),
        (asw, hmsk, [('outputnode.bias_corrected', 'inputnode.in_file')]),
        (segment, hmsk, [('tissue_class_map', 'inputnode.in_segm')]),
        (asw, norm, [('outputnode.bias_corrected', 'inputnode.moving_image'),
                     ('outputnode.out_mask', 'inputnode.moving_mask')]),
        (norm, amw, [
            ('outputnode.inverse_composite_transform', 'inputnode.inverse_composite_transform')]),
        (norm, iqmswf, [
            ('outputnode.inverse_composite_transform', 'inputnode.inverse_composite_transform')]),
        (norm, repwf, ([
            ('outputnode.out_report', 'inputnode.mni_report')])),
        (to_ras, amw, [('out_file', 'inputnode.in_file')]),
        (asw, amw, [('outputnode.out_mask', 'inputnode.in_mask')]),
        (hmsk, amw, [('outputnode.out_file', 'inputnode.head_mask')]),
        (to_ras, iqmswf, [('out_file', 'inputnode.in_ras')]),
        (asw, iqmswf, [('outputnode.bias_corrected', 'inputnode.inu_corrected'),
                       ('outputnode.bias_image', 'inputnode.in_inu'),
                       ('outputnode.out_mask', 'inputnode.brainmask')]),
        (amw, iqmswf, [('outputnode.out_file', 'inputnode.airmask'),
                       ('outputnode.artifact_msk', 'inputnode.artmask'),
                       ('outputnode.rot_mask', 'inputnode.rotmask')]),
        (segment, iqmswf, [('tissue_class_map', 'inputnode.segmentation'),
                           ('partial_volume_files', 'inputnode.pvms')]),
        (hmsk, iqmswf, [('outputnode.out_file', 'inputnode.headmask')]),
        (to_ras, repwf, [('out_file', 'inputnode.in_ras')]),
        (asw, repwf, [('outputnode.bias_corrected', 'inputnode.inu_corrected'),
                      ('outputnode.out_mask', 'inputnode.brainmask')]),
        (hmsk, repwf, [('outputnode.out_file', 'inputnode.headmask')]),
        (amw, repwf, [('outputnode.out_file', 'inputnode.airmask'),
                      ('outputnode.artifact_msk', 'inputnode.artmask'),
                      ('outputnode.rot_mask', 'inputnode.rotmask')]),
        (segment, repwf, [('tissue_class_map', 'inputnode.segmentation')]),
        (iqmswf, repwf, [('outputnode.out_noisefit', 'inputnode.noisefit')]),
        (iqmswf, repwf, [('outputnode.out_file', 'inputnode.in_iqms')]),
        (iqmswf, outputnode, [('outputnode.out_file', 'out_json')])
    ])

    # Upload metrics
    if not settings.get('no_sub', False):
        from ..interfaces.webapi import UploadIQMs
        upldwf = pe.Node(UploadIQMs(), name='UploadMetrics')
        upldwf.inputs.email = settings.get('email', '')
        upldwf.inputs.url = settings.get('webapi_url')
        if settings.get('webapi_port'):
            upldwf.inputs.port = settings.get('webapi_port')

        upldwf.inputs.strict = settings.get('upload_strict', False)

        workflow.connect([
            (iqmswf, upldwf, [('outputnode.out_file', 'in_iqms')]),
        ])

    return workflow

def spatial_normalization(settings, mod='T1w', name='SpatialNormalization',
                          resolution=2.0):
    """
    A simple workflow to perform spatial normalization

    """
    from niworkflows.data import getters as niwgetters

    # Have some settings handy
    tpl_id = settings.get('template_id', 'mni_icbm152_nlin_asym_09c')
    mni_template = getattr(niwgetters, 'get_{}'.format(tpl_id))()

    # Define workflow interface
    workflow = pe.Workflow(name=name)
    inputnode = pe.Node(niu.IdentityInterface(fields=[
        'moving_image', 'moving_mask']), name='inputnode')
    outputnode = pe.Node(niu.IdentityInterface(fields=[
        'inverse_composite_transform', 'out_report']), name='outputnode')

    # Spatial normalization
    norm = pe.Node(RobustMNINormalization(
        flavor='testing' if settings.get('testing', False) else 'fast',
        num_threads=settings.get('ants_nthreads'),
        template=tpl_id,
        template_resolution=2,
        reference=mod[:2],
        generate_report=True,),
                   name='SpatialNormalization',
                   # Request all MultiProc processes when ants_nthreads > n_procs
                   num_threads=min(settings.get('ants_nthreads', DEFAULTS['ants_nthreads']),
                                   settings.get('n_procs', 1)),
                   estimated_memory_gb=3)
    norm.inputs.reference_mask = op.join(mni_template,
                                     '%dmm_brainmask.nii.gz' % int(resolution))

    workflow.connect([
        (inputnode, norm, [('moving_image', 'moving_image'),
                           ('moving_mask', 'moving_mask')]),
        (norm, outputnode, [('inverse_composite_transform', 'inverse_composite_transform'),
                            ('out_report', 'out_report')]),
    ])
    return workflow

def compute_iqms(settings, modality='T1w', name='ComputeIQMs'):
    """
    Workflow that actually computes the IQMs

    .. workflow::

        from mriqc.workflows.anatomical import compute_iqms
        wf = compute_iqms(settings={'output_dir': 'out'})

    """
    from .utils import _tofloat
    from ..interfaces.anatomical import Harmonize

    workflow = pe.Workflow(name=name)
    inputnode = pe.Node(niu.IdentityInterface(fields=[
        'in_file', 'in_ras',
        'brainmask', 'airmask', 'artmask', 'headmask', 'rotmask',
        'segmentation', 'inu_corrected', 'in_inu', 'pvms', 'metadata',
        'inverse_composite_transform']), name='inputnode')
    outputnode = pe.Node(niu.IdentityInterface(fields=['out_file', 'out_noisefit']),
                         name='outputnode')

    deriv_dir = check_folder(op.abspath(op.join(settings['output_dir'], 'derivatives')))

    # Extract metadata
    meta = pe.Node(ReadSidecarJSON(), name='metadata')

    # Add provenance
    addprov = pe.Node(niu.Function(function=_add_provenance), name='provenance')
    addprov.inputs.settings = {
        'testing': settings.get('testing', False)
    }

    # AFNI check smoothing
    fwhm = pe.Node(afni.FWHMx(combine=True, detrend=True), name='smoothness')
    # fwhm.inputs.acf = True  # add when AFNI >= 16

    # Harmonize
    homog = pe.Node(Harmonize(), name='harmonize')

    # Mortamet's QI2
    getqi2 = pe.Node(ComputeQI2(erodemsk=settings.get('testing', False)),
                     name='ComputeQI2')

    # Compute python-coded measures
    measures = pe.Node(StructuralQC(), 'measures')

    # Project MNI segmentation to T1 space
    invt = pe.MapNode(ants.ApplyTransforms(
        dimension=3, default_value=0, interpolation='Linear',
        float=True),
        iterfield=['input_image'], name='MNItpms2t1')
    invt.inputs.input_image = [op.join(get_mni_icbm152_nlin_asym_09c(), fname + '.nii.gz')
                               for fname in ['1mm_tpm_csf', '1mm_tpm_gm', '1mm_tpm_wm']]

    datasink = pe.Node(IQMFileSink(modality=modality, out_dir=deriv_dir),
                       name='datasink')
    datasink.inputs.modality = modality

    def _getwm(inlist):
        return inlist[-1]

    workflow.connect([
        (inputnode, meta, [('in_file', 'in_file')]),
        (meta, datasink, [('subject_id', 'subject_id'),
                          ('session_id', 'session_id'),
                          ('acq_id', 'acq_id'),
                          ('rec_id', 'rec_id'),
                          ('run_id', 'run_id'),
                          ('out_dict', 'metadata')]),

        (inputnode, addprov, [('in_file', 'in_file'),
                              ('airmask', 'air_msk'),
                              ('rotmask', 'rot_msk')]),
        (inputnode, getqi2, [('in_ras', 'in_file'),
                             ('airmask', 'air_msk')]),
        (inputnode, homog, [('inu_corrected', 'in_file'),
                            (('pvms', _getwm), 'wm_mask')]),
        (inputnode, measures, [('in_inu', 'in_bias'),
                               ('in_ras', 'in_file'),
                               ('airmask', 'air_msk'),
                               ('headmask', 'head_msk'),
                               ('artmask', 'artifact_msk'),
                               ('rotmask', 'rot_msk'),
                               ('segmentation', 'in_segm'),
                               ('pvms', 'in_pvms')]),
        (inputnode, fwhm, [('in_ras', 'in_file'),
                           ('brainmask', 'mask')]),
        (inputnode, invt, [('in_ras', 'reference_image'),
                           ('inverse_composite_transform', 'transforms')]),
        (homog, measures, [('out_file', 'in_noinu')]),
        (invt, measures, [('output_image', 'mni_tpms')]),
        (fwhm, measures, [(('fwhm', _tofloat), 'in_fwhm')]),
        (measures, datasink, [('out_qc', 'root')]),
        (addprov, datasink, [('out', 'provenance')]),
        (getqi2, datasink, [('qi2', 'qi_2')]),
        (getqi2, outputnode, [('out_file', 'out_noisefit')]),
        (datasink, outputnode, [('out_file', 'out_file')]),
    ])
    return workflow

def individual_reports(settings, name='ReportsWorkflow'):
    """
    Encapsulates nodes writing plots

    .. workflow::

        from mriqc.workflows.anatomical import individual_reports
        wf = individual_reports(settings={'output_dir': 'out'})

    """
    from ..interfaces import PlotMosaic
    from ..reports import individual_html

    verbose = settings.get('verbose_reports', False)
    pages = 2
    extra_pages = 0
    if verbose:
        extra_pages = 7

    workflow = pe.Workflow(name=name)
    inputnode = pe.Node(niu.IdentityInterface(fields=[
        'in_ras', 'brainmask', 'headmask', 'airmask', 'artmask', 'rotmask',
        'segmentation', 'inu_corrected', 'noisefit', 'in_iqms',
        'mni_report']),
        name='inputnode')

    mosaic_zoom = pe.Node(PlotMosaic(
        out_file='plot_anat_mosaic1_zoomed.svg',
        title='zoomed',
        cmap='Greys_r'), name='PlotMosaicZoomed')

    mosaic_noise = pe.Node(PlotMosaic(
        out_file='plot_anat_mosaic2_noise.svg',
        title='noise enhanced',
        only_noise=True,
        cmap='viridis_r'), name='PlotMosaicNoise')

    mplots = pe.Node(niu.Merge(pages + extra_pages), name='MergePlots')
    rnode = pe.Node(niu.Function(
        input_names=['in_iqms', 'in_plots'], output_names=['out_file'],
        function=individual_html), name='GenerateReport')

    # Link images that should be reported
    dsplots = pe.Node(nio.DataSink(
        base_directory=settings['output_dir'], parameterization=False), name='dsplots')
    dsplots.inputs.container = 'reports'

    workflow.connect([
        (inputnode, rnode, [('in_iqms', 'in_iqms')]),
        (inputnode, mosaic_zoom, [('in_ras', 'in_file'),
                                  ('brainmask', 'bbox_mask_file')]),
        (inputnode, mosaic_noise, [('in_ras', 'in_file')]),
        (mosaic_zoom, mplots, [('out_file', "in1")]),
        (mosaic_noise, mplots, [('out_file', "in2")]),
        (mplots, rnode, [('out', 'in_plots')]),
        (rnode, dsplots, [('out_file', "@html_report")]),
    ])

    if not verbose:
        return workflow

    from ..interfaces.viz import PlotContours
    from ..viz.utils import plot_bg_dist
    plot_bgdist = pe.Node(niu.Function(input_names=['in_file'], output_names=['out_file'],
                          function=plot_bg_dist), name='PlotBackground')

    plot_segm = pe.Node(PlotContours(
        display_mode='z', levels=[.5, 1.5, 2.5], cut_coords=10,
        colors=['r', 'g', 'b']), name='PlotSegmentation')

    plot_bmask = pe.Node(PlotContours(
        display_mode='z', levels=[.5], colors=['r'], cut_coords=10,
        out_file='bmask'), name='PlotBrainmask')
    plot_airmask = pe.Node(PlotContours(
        display_mode='x', levels=[.5], colors=['r'],
        cut_coords=6, out_file='airmask'), name='PlotAirmask')
    plot_headmask = pe.Node(PlotContours(
        display_mode='x', levels=[.5], colors=['r'],
        cut_coords=6, out_file='headmask'), name='PlotHeadmask')
    plot_artmask = pe.Node(PlotContours(
        display_mode='z', levels=[.5], colors=['r'], cut_coords=10,
        out_file='artmask', saturate=True), name='PlotArtmask')

    workflow.connect([
        (inputnode, plot_segm, [('in_ras', 'in_file'),
                                ('segmentation', 'in_contours')]),
        (inputnode, plot_bmask, [('in_ras', 'in_file'),
                                 ('brainmask', 'in_contours')]),
        (inputnode, plot_headmask, [('in_ras', 'in_file'),
                                    ('headmask', 'in_contours')]),
        (inputnode, plot_airmask, [('in_ras', 'in_file'),
                                   ('airmask', 'in_contours')]),
        (inputnode, plot_artmask, [('in_ras', 'in_file'),
                                   ('artmask', 'in_contours')]),
        (inputnode, plot_bgdist, [('noisefit', 'in_file')]),
        (inputnode, mplots, [('mni_report', "in%d" % (pages + 1))]),
        (plot_bmask, mplots, [('out_file', 'in%d' % (pages + 2))]),
        (plot_segm, mplots, [('out_file', 'in%d' % (pages + 3))]),
        (plot_artmask, mplots, [('out_file', 'in%d' % (pages + 4))]),
        (plot_headmask, mplots, [('out_file', 'in%d' % (pages + 5))]),
        (plot_airmask, mplots, [('out_file', 'in%d' % (pages + 6))]),
        (plot_bgdist, mplots, [('out_file', 'in%d' % (pages + 7))])
    ])
    return workflow

def headmsk_wf(name='HeadMaskWorkflow', use_bet=True):
    """
    Computes a head mask as in [Mortamet2009]_.

    .. workflow::

        from mriqc.workflows.anatomical import headmsk_wf
        wf = headmsk_wf()

    """

    has_dipy = False
    try:
        from dipy.denoise import nlmeans
        has_dipy = True
    except ImportError:
        pass

    workflow = pe.Workflow(name=name)
    inputnode = pe.Node(niu.IdentityInterface(fields=['in_file', 'in_segm']),
                        name='inputnode')
    outputnode = pe.Node(niu.IdentityInterface(fields=['out_file']), name='outputnode')

    if use_bet or not has_dipy:
        # Alternative for when dipy is not installed
        bet = pe.Node(fsl.BET(surfaces=True), name='fsl_bet')
        workflow.connect([
            (inputnode, bet, [('in_file', 'in_file')]),
            (bet, outputnode, [('outskin_mask_file', 'out_file')])
        ])

    else:
        from niworkflows.nipype.interfaces.dipy import Denoise
        enhance = pe.Node(niu.Function(
            input_names=['in_file'], output_names=['out_file'], function=_enhance), name='Enhance')
        estsnr = pe.Node(niu.Function(
            input_names=['in_file', 'seg_file'], output_names=['out_snr'],
            function=_estimate_snr), name='EstimateSNR')
        denoise = pe.Node(Denoise(), name='Denoise')
        gradient = pe.Node(niu.Function(
            input_names=['in_file', 'snr'], output_names=['out_file'], function=image_gradient), name='Grad')
        thresh = pe.Node(niu.Function(
            input_names=['in_file', 'in_segm'], output_names=['out_file'], function=gradient_threshold),
                         name='GradientThreshold')

        workflow.connect([
            (inputnode, estsnr, [('in_file', 'in_file'),
                                 ('in_segm', 'seg_file')]),
            (estsnr, denoise, [('out_snr', 'snr')]),
            (inputnode, enhance, [('in_file', 'in_file')]),
            (enhance, denoise, [('out_file', 'in_file')]),
            (estsnr, gradient, [('out_snr', 'snr')]),
            (denoise, gradient, [('out_file', 'in_file')]),
            (inputnode, thresh, [('in_segm', 'in_segm')]),
            (gradient, thresh, [('out_file', 'in_file')]),
            (thresh, outputnode, [('out_file', 'out_file')])
        ])

    return workflow


def airmsk_wf(name='AirMaskWorkflow'):
    """
    Implements the Step 1 of [Mortamet2009]_.

    .. workflow::

        from mriqc.workflows.anatomical import airmsk_wf
        wf = airmsk_wf()

    """
    workflow = pe.Workflow(name=name)

    inputnode = pe.Node(niu.IdentityInterface(
        fields=['in_file', 'in_mask', 'head_mask', 'inverse_composite_transform']),
        name='inputnode')
    outputnode = pe.Node(niu.IdentityInterface(fields=['out_file', 'artifact_msk', 'rot_mask']),
                         name='outputnode')

    rotmsk = pe.Node(RotationMask(), name='RotationMask')

    invt = pe.Node(ants.ApplyTransforms(dimension=3, default_value=0,
                                        interpolation='Linear', float=True), name='invert_xfm')
    invt.inputs.input_image = op.join(get_mni_icbm152_nlin_asym_09c(), '1mm_headmask.nii.gz')

    binarize = pe.Node(niu.Function(function=_binarize), name='Binarize')

    qi1 = pe.Node(ArtifactMask(), name='ArtifactMask')

    workflow.connect([
        (inputnode, rotmsk, [('in_file', 'in_file')]),
        (inputnode, qi1, [('in_file', 'in_file'),
                          ('head_mask', 'head_mask')]),
        (rotmsk, qi1, [('out_file', 'rot_mask')]),
        (inputnode, invt, [('in_mask', 'reference_image'),
                           ('inverse_composite_transform', 'transforms')]),
        (invt, binarize, [('output_image', 'in_file')]),
        (binarize, qi1, [('out', 'nasion_post_mask')]),
        (qi1, outputnode, [('out_air_msk', 'out_file'),
                           ('out_art_msk', 'artifact_msk')]),
        (rotmsk, outputnode, [('out_file', 'rot_mask')])
    ])
    return workflow


def _add_provenance(in_file, settings, air_msk, rot_msk):
    from mriqc import __version__ as version
    from niworkflows.nipype.utils.filemanip import hash_infile
    import nibabel as nb
    import numpy as np

    air_msk_size = nb.load(air_msk).get_data().astype(
        np.uint8).sum()
    rot_msk_size = nb.load(rot_msk).get_data().astype(
        np.uint8).sum()

    out_prov = {
        'md5sum': hash_infile(in_file),
        'version': version,
        'software': 'mriqc',
        'warnings': {
            'small_air_mask': bool(air_msk_size < 5e5),
            'large_rot_frame': bool(rot_msk_size > 500),
        }
    }

    if settings:
        out_prov['settings'] = settings

    return out_prov

def _binarize(in_file, threshold=0.5, out_file=None):
    import os.path as op
    import numpy as np
    import nibabel as nb

    if out_file is None:
        fname, ext = op.splitext(op.basename(in_file))
        if ext == '.gz':
            fname, ext2 = op.splitext(fname)
            ext = ext2 + ext
        out_file = op.abspath('{}_bin{}'.format(fname, ext))

    nii = nb.load(in_file)
    data = nii.get_data()

    data[data <= threshold] = 0
    data[data > 0] = 1

    hdr = nii.header.copy()
    hdr.set_data_dtype(np.uint8)
    nb.Nifti1Image(data.astype(np.uint8), nii.affine, hdr).to_filename(
        out_file)
    return out_file

def _estimate_snr(in_file, seg_file):
    import nibabel as nb
    from mriqc.qc.anatomical import snr
    out_snr = snr(nb.load(in_file).get_data(), nb.load(seg_file).get_data(),
                  fglabel='wm')
    return out_snr

def _enhance(in_file, out_file=None):
    import os.path as op
    import numpy as np
    import nibabel as nb

    if out_file is None:
        fname, ext = op.splitext(op.basename(in_file))
        if ext == '.gz':
            fname, ext2 = op.splitext(fname)
            ext = ext2 + ext
        out_file = op.abspath('{}_enhanced{}'.format(fname, ext))

    imnii = nb.load(in_file)
    data = imnii.get_data().astype(np.float32)  # pylint: disable=no-member
    range_max = np.percentile(data[data > 0], 99.98)
    range_min = np.median(data[data > 0])

    # Resample signal excess pixels
    excess = np.where(data > range_max)
    data[excess] = 0
    data[excess] = np.random.choice(data[data > range_min], size=len(excess[0]))

    nb.Nifti1Image(data, imnii.get_affine(), imnii.get_header()).to_filename(
        out_file)

    return out_file

def image_gradient(in_file, snr, out_file=None):
    """Computes the magnitude gradient of an image using numpy"""
    import os.path as op
    import numpy as np
    import nibabel as nb
    from scipy.ndimage import gaussian_gradient_magnitude as gradient

    if out_file is None:
        fname, ext = op.splitext(op.basename(in_file))
        if ext == '.gz':
            fname, ext2 = op.splitext(fname)
            ext = ext2 + ext
        out_file = op.abspath('{}_grad{}'.format(fname, ext))

    imnii = nb.load(in_file)
    data = imnii.get_data().astype(np.float32)  # pylint: disable=no-member
    datamax = np.percentile(data.reshape(-1), 99.5)
    data *= 100 / datamax
    grad = gradient(data, 3.0)
    gradmax = np.percentile(grad.reshape(-1), 99.5)
    grad *= 100.
    grad /= gradmax

    nb.Nifti1Image(grad, imnii.get_affine(), imnii.get_header()).to_filename(out_file)
    return out_file

def gradient_threshold(in_file, in_segm, thresh=1.0, out_file=None):
    """ Compute a threshold from the histogram of the magnitude gradient image """
    import os.path as op
    import numpy as np
    import nibabel as nb
    from scipy import ndimage as sim

    struc = sim.iterate_structure(sim.generate_binary_structure(3, 2), 2)

    if out_file is None:
        fname, ext = op.splitext(op.basename(in_file))
        if ext == '.gz':
            fname, ext2 = op.splitext(fname)
            ext = ext2 + ext
        out_file = op.abspath('{}_gradmask{}'.format(fname, ext))

    imnii = nb.load(in_file)

    hdr = imnii.get_header().copy()
    hdr.set_data_dtype(np.uint8)  # pylint: disable=no-member

    data = imnii.get_data().astype(np.float32)

    mask = np.zeros_like(data, dtype=np.uint8)  # pylint: disable=no-member
    mask[data > 15.] = 1

    segdata = nb.load(in_segm).get_data().astype(np.uint8)
    segdata[segdata > 0] = 1
    segdata = sim.binary_dilation(segdata, struc, iterations=2, border_value=1).astype(np.uint8)  # pylint: disable=no-member
    mask[segdata > 0] = 1

    mask = sim.binary_closing(mask, struc, iterations=2).astype(np.uint8)  # pylint: disable=no-member
    # Remove small objects
    label_im, nb_labels = sim.label(mask)
    artmsk = np.zeros_like(mask)
    if nb_labels > 2:
        sizes = sim.sum(mask, label_im, list(range(nb_labels + 1)))
        ordered = list(reversed(sorted(zip(sizes, list(range(nb_labels + 1))))))
        for _, label in ordered[2:]:
            mask[label_im == label] = 0
            artmsk[label_im == label] = 1

    mask = sim.binary_fill_holes(mask, struc).astype(np.uint8)  # pylint: disable=no-member

    nb.Nifti1Image(mask, imnii.get_affine(), hdr).to_filename(out_file)
    return out_file
