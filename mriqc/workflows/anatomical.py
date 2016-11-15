#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# @Author: oesteban
# @Date:   2016-01-05 11:24:05
# @Email:  code@oscaresteban.es
# @Last modified by:   oesteban
# @Last Modified time: 2016-11-15 09:50:23
""" A QC workflow for anatomical MRI """
from __future__ import print_function, division, absolute_import, unicode_literals
from builtins import zip, range
import os.path as op

from nipype.pipeline import engine as pe
from nipype.interfaces import io as nio
from nipype.interfaces import utility as niu
from nipype.interfaces import fsl
from nipype.interfaces import ants
from nipype.interfaces import afni

from niworkflows.data import get_mni_icbm152_nlin_asym_09c
from niworkflows.anat.skullstrip import afni_wf as skullstrip_wf
from niworkflows.anat.mni import RobustMNINormalization
from mriqc.workflows.utils import fwhm_dict
from mriqc.interfaces import (StructuralQC, ArtifactMask, ReadSidecarJSON,
                              ConformImage, ComputeQI2)

from mriqc.utils.misc import bids_path, check_folder


def anat_qc_workflow(dataset, settings, name='anatMRIQC'):
    """
    One-subject-one-session-one-run pipeline to extract the NR-IQMs from
    anatomical images
    """

    workflow = pe.Workflow(name=name)

    # Define workflow, inputs and outputs
    # 0. Get data
    inputnode = pe.Node(niu.IdentityInterface(fields=['in_file']), name='inputnode')
    inputnode.iterables = [('in_file', dataset)]

    outputnode = pe.Node(niu.IdentityInterface(fields=['out_json']), name='outputnode')

    meta = pe.Node(ReadSidecarJSON(), name='metadata')

    # 1a. Reorient anatomical image
    to_ras = pe.Node(ConformImage(), name='conform')
    # 1b. Estimate bias
    n4itk = pe.Node(ants.N4BiasFieldCorrection(dimension=3, save_bias=True), name='Bias')
    # 2. Skull-stripping (afni)
    asw = skullstrip_wf()
    # 3. Head mask (including nasial-cerebelum mask)
    hmsk = headmsk_wf()
    # 4. Spatial Normalization, using ANTs
    norm = pe.Node(RobustMNINormalization(
        num_threads=settings.get('ants_nthreads', 6), template='mni_icbm152_nlin_asym_09c',
        testing=settings.get('testing', False)), name='SpatialNormalization')
    # 5. Air mask (with and without artifacts)
    amw = airmsk_wf()
    # 6. Brain tissue segmentation
    segment = pe.Node(fsl.FAST(
        img_type=1, segments=True, out_basename='segment'), name='segmentation')
    # 7. Compute IQMs
    iqmswf = compute_iqms(settings)
    # Reports
    repwf = individual_reports(settings)

    # Connect all nodes
    workflow.connect([
        (inputnode, to_ras, [('in_file', 'in_file')]),
        (inputnode, meta, [('in_file', 'in_file')]),
        (to_ras, n4itk, [('out_file', 'input_image')]),
        (meta, iqmswf, [('subject_id', 'inputnode.subject_id'),
                        ('session_id', 'inputnode.session_id'),
                        ('run_id', 'inputnode.run_id')]),
        (meta, repwf, [('subject_id', 'inputnode.subject_id'),
                       ('session_id', 'inputnode.session_id'),
                       ('run_id', 'inputnode.run_id'),
                       ('out_dict', 'inputnode.in_metadata')]),
        (n4itk, asw, [('output_image', 'inputnode.in_file')]),
        (asw, segment, [('outputnode.out_file', 'in_files')]),
        (n4itk, hmsk, [('output_image', 'inputnode.in_file')]),
        (segment, hmsk, [('tissue_class_map', 'inputnode.in_segm')]),
        (n4itk, norm, [('output_image', 'moving_image')]),
        (asw, norm, [('outputnode.out_mask', 'moving_mask')]),
        (to_ras, amw, [('out_file', 'inputnode.in_file')]),
        (norm, amw, [('reverse_transforms', 'inputnode.reverse_transforms'),
                     ('reverse_invert_flags', 'inputnode.reverse_invert_flags')]),
        (norm, iqmswf, [('reverse_transforms', 'inputnode.reverse_transforms'),
                     ('reverse_invert_flags', 'inputnode.reverse_invert_flags')]),
        (asw, amw, [('outputnode.out_mask', 'inputnode.in_mask')]),
        (hmsk, amw, [('outputnode.out_file', 'inputnode.head_mask')]),
        (to_ras, iqmswf, [('out_file', 'inputnode.orig')]),
        (n4itk, iqmswf, [('output_image', 'inputnode.inu_corrected'),
                         ('bias_image', 'inputnode.in_inu')]),
        (asw, iqmswf, [('outputnode.out_mask', 'inputnode.brainmask')]),
        (amw, iqmswf, [('outputnode.out_file', 'inputnode.airmask'),
                       ('outputnode.artifact_msk', 'inputnode.artmask')]),
        (segment, iqmswf, [('tissue_class_map', 'inputnode.segmentation'),
                           ('partial_volume_files', 'inputnode.pvms')]),
        (meta, iqmswf, [('out_dict', 'inputnode.metadata')]),
        (hmsk, iqmswf, [('outputnode.out_file', 'inputnode.headmask')]),
        (to_ras, repwf, [('out_file', 'inputnode.orig')]),
        (n4itk, repwf, [('output_image', 'inputnode.inu_corrected')]),
        (asw, repwf, [('outputnode.out_mask', 'inputnode.brainmask')]),
        (hmsk, repwf, [('outputnode.out_file', 'inputnode.headmask')]),
        (amw, repwf, [('outputnode.out_file', 'inputnode.airmask'),
                      ('outputnode.artifact_msk', 'inputnode.artmask')]),
        (segment, repwf, [('tissue_class_map', 'inputnode.segmentation')]),
        (iqmswf, repwf, [('outputnode.out_noisefit', 'inputnode.noisefit')]),
        (iqmswf, repwf, [('outputnode.out_file', 'inputnode.in_iqms')]),
        (iqmswf, outputnode, [('outputnode.out_file', 'out_json')])
    ])

    return workflow

def compute_iqms(settings, name='ComputeIQMs'):
    """Workflow that actually computes the IQMs"""
    workflow = pe.Workflow(name=name)
    inputnode = pe.Node(niu.IdentityInterface(fields=[
        'subject_id', 'session_id', 'run_id', 'orig', 'brainmask', 'airmask', 'artmask',
        'headmask', 'segmentation', 'inu_corrected', 'in_inu', 'pvms', 'metadata',
        'reverse_transforms', 'reverse_invert_flags']), name='inputnode')
    outputnode = pe.Node(niu.IdentityInterface(fields=['out_file', 'out_noisefit']),
                         name='outputnode')

    deriv_dir = check_folder(op.abspath(op.join(settings['output_dir'], 'derivatives')))

    # AFNI check smoothing
    fwhm = pe.Node(afni.FWHMx(combine=True, detrend=True), name='smoothness')
    # fwhm.inputs.acf = True  # add when AFNI >= 16

    # Mortamet's QI2
    getqi2 = pe.Node(ComputeQI2(erodemsk=settings.get('testing', False)),
                     name='ComputeQI2')

    # Compute python-coded measures
    measures = pe.Node(StructuralQC(), 'measures')

    # Project MNI segmentation to T1 space
    invt = pe.MapNode(ants.ApplyTransforms(
        dimension=3, default_value=0, interpolation='NearestNeighbor'),
        iterfield=['input_image'], name='MNItpms2t1')
    invt.inputs.input_image = [op.join(get_mni_icbm152_nlin_asym_09c(), fname + '.nii.gz')
                               for fname in ['1mm_tpm_csf', '1mm_tpm_gm', '1mm_tpm_wm']]

    # Format name
    out_name = pe.Node(niu.Function(
        input_names=['subid', 'sesid', 'runid', 'prefix', 'out_path'], output_names=['out_file'],
        function=bids_path), name='FormatName')
    out_name.inputs.out_path = deriv_dir
    out_name.inputs.prefix = 'anat'

    # Save to JSON file
    jfs_if = nio.JSONFileSink()
    setattr(jfs_if, '_always_run', settings.get('force_run', False))
    datasink = pe.Node(jfs_if, name='datasink')
    datasink.inputs.qc_type = 'anat'

    workflow.connect([
        (inputnode, out_name, [('subject_id', 'subid'),
                               ('session_id', 'sesid'),
                               ('run_id', 'runid')]),
        (inputnode, datasink, [('subject_id', 'subject_id'),
                               ('session_id', 'session_id'),
                               ('run_id', 'run_id'),
                               ('metadata', 'metadata')]),
        (inputnode, getqi2, [('orig', 'in_file'),
                             ('airmask', 'air_msk')]),
        (inputnode, measures, [('inu_corrected', 'in_noinu'),
                               ('in_inu', 'in_bias'),
                               ('orig', 'in_file'),
                               ('airmask', 'air_msk'),
                               ('headmask', 'head_msk'),
                               ('artmask', 'artifact_msk'),
                               ('segmentation', 'in_segm'),
                               ('pvms', 'in_pvms')]),
        (inputnode, fwhm, [('orig', 'in_file'),
                           ('brainmask', 'mask')]),
        (inputnode, invt, [('orig', 'reference_image'),
                           ('reverse_transforms', 'transforms'),
                           ('reverse_invert_flags', 'invert_transform_flags')]),
        (invt, measures, [('output_image', 'mni_tpms')]),
        (fwhm, datasink, [(('fwhm', fwhm_dict), 'fwhm')]),
        (getqi2, datasink, [('qi2', 'qi2')]),
        (measures, datasink, [('summary', 'summary'),
                              ('spacing', 'spacing'),
                              ('size', 'size'),
                              ('icvs', 'icvs'),
                              ('rpve', 'rpve'),
                              ('inu', 'inu'),
                              ('snr', 'snr'),
                              ('cnr', 'cnr'),
                              ('fber', 'fber'),
                              ('efc', 'efc'),
                              ('qi1', 'qi1'),
                              ('cjv', 'cjv'),
                              ('wm2max', 'wm2max'),
                              ('tpm_overlap', 'tpm_overlap')]),
        (out_name, datasink, [('out_file', 'out_file')]),
        (getqi2, outputnode, [('out_file', 'out_noisefit')]),
        (datasink, outputnode, [('out_file', 'out_file')])
    ])
    return workflow


def individual_reports(settings, name='ReportsWorkflow'):
    """Encapsulates nodes writing plots"""
    from mriqc.interfaces import PlotMosaic
    from mriqc.reports import individual_html

    verbose = settings.get('verbose_reports', False)
    pages = 2
    if verbose:
        pages += 6

    workflow = pe.Workflow(name=name)
    inputnode = pe.Node(niu.IdentityInterface(fields=[
        'subject_id', 'session_id', 'run_id', 'in_metadata',
        'orig', 'brainmask', 'headmask', 'airmask', 'artmask',
        'segmentation', 'inu_corrected', 'noisefit', 'in_iqms']),
        name='inputnode')

    # T1w mosaic plot
    mosaic_zoom = pe.Node(PlotMosaic(
        out_file='plot_anat_mosaic1_zoomed.svg',
        title='T1w (zoomed) session: {session_id} run: {run_id}',
        cmap='Greys_r'), name='PlotMosaicZoomed')

    mosaic_noise = pe.Node(PlotMosaic(
        out_file='plot_anat_mosaic2_noise.svg',
        title='T1w (noise) session: {session_id} run: {run_id}',
        only_noise=True,
        cmap='viridis_r'), name='PlotMosaicNoise')

    mplots = pe.Node(niu.Merge(pages), name='MergePlots')
    rnode = pe.Node(niu.Function(
        input_names=['in_iqms', 'in_metadata', 'in_plots'], output_names=['out_file'],
        function=individual_html), name='GenerateReport')

    # Link images that should be reported
    dsplots = pe.Node(nio.DataSink(
        base_directory=settings['output_dir'], parameterization=False), name='dsplots')
    dsplots.inputs.container = 'reports'

    workflow.connect([
        (inputnode, rnode, [('in_iqms', 'in_iqms'),
                            ('in_metadata', 'in_metadata')]),
        (inputnode, mosaic_zoom, [('subject_id', 'subject_id'),
                                  ('session_id', 'session_id'),
                                  ('run_id', 'run_id'),
                                  ('orig', 'in_file'),
                                  ('brainmask', 'bbox_mask_file')]),

        (inputnode, mosaic_noise, [('subject_id', 'subject_id'),
                                   ('session_id', 'session_id'),
                                   ('run_id', 'run_id'),
                                   ('orig', 'in_file')]),
        (mosaic_zoom, mplots, [('out_file', "in1")]),
        (mosaic_noise, mplots, [('out_file', "in2")]),
        (mplots, rnode, [('out', 'in_plots')]),
        (rnode, dsplots, [('out_file', "@html_report")]),
    ])

    if not verbose:
        return workflow

    from mriqc.interfaces.viz import PlotContours
    from mriqc.interfaces.viz_utils import plot_bg_dist
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
        (inputnode, plot_segm, [('orig', 'in_file'),
                                ('segmentation', 'in_contours')]),
        (inputnode, plot_bmask, [('orig', 'in_file'),
                                 ('brainmask', 'in_contours')]),
        (inputnode, plot_headmask, [('orig', 'in_file'),
                                    ('headmask', 'in_contours')]),
        (inputnode, plot_airmask, [('orig', 'in_file'),
                                   ('airmask', 'in_contours')]),
        (inputnode, plot_artmask, [('orig', 'in_file'),
                                   ('artmask', 'in_contours')]),
        (inputnode, plot_bgdist, [('noisefit', 'in_file')]),

        (plot_bmask, mplots, [('out_file', 'in3')]),
        (plot_segm, mplots, [('out_file', 'in4')]),
        (plot_artmask, mplots, [('out_file', 'in5')]),
        (plot_headmask, mplots, [('out_file', 'in6')]),
        (plot_airmask, mplots, [('out_file', 'in7')]),
        (plot_bgdist, mplots, [('out_file', 'in8')])
    ])
    return workflow

def headmsk_wf(name='HeadMaskWorkflow', use_bet=True):
    """Computes a head mask as in [Mortamet2009]_."""

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


    enhance = pe.Node(niu.Function(
        input_names=['in_file'], output_names=['out_file'], function=_enhance), name='Enhance')


    if use_bet or not has_dipy:
        # Alternative for when dipy is not installed
        bet = pe.Node(fsl.BET(surfaces=True), name='fsl_bet')
        workflow.connect([
            (inputnode, enhance, [('in_file', 'in_file')]),
            (enhance, bet, [('out_file', 'in_file')]),
            (bet, outputnode, [('outskin_mask_file', 'out_file')])
        ])

    else:
        from nipype.interfaces.dipy import Denoise
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
    """Implements the Step 1 of [Mortamet2009]_."""
    workflow = pe.Workflow(name=name)

    inputnode = pe.Node(niu.IdentityInterface(
        fields=['in_file', 'in_mask', 'head_mask', 'reverse_transforms', 'reverse_invert_flags']),
        name='inputnode')
    outputnode = pe.Node(niu.IdentityInterface(fields=['out_file', 'artifact_msk']),
                         name='outputnode')

    invt = pe.Node(ants.ApplyTransforms(
        dimension=3, default_value=0, interpolation='NearestNeighbor'), name='invert_xfm')
    invt.inputs.input_image = op.join(get_mni_icbm152_nlin_asym_09c(), '1mm_headmask.nii.gz')

    qi1 = pe.Node(ArtifactMask(), name='ArtifactMask')

    workflow.connect([
        (inputnode, qi1, [('in_file', 'in_file'),
                          ('head_mask', 'head_mask')]),
        (inputnode, invt, [('in_mask', 'reference_image'),
                           ('reverse_transforms', 'transforms'),
                           ('reverse_invert_flags', 'invert_transform_flags')]),
        (invt, qi1, [('output_image', 'nasion_post_mask')]),
        (qi1, outputnode, [('out_air_msk', 'out_file'),
                           ('out_art_msk', 'artifact_msk')])
    ])
    return workflow

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
