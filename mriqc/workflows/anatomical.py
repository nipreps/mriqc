#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# @Author: oesteban
# @Date:   2016-01-05 11:24:05
# @Email:  code@oscaresteban.es
# @Last modified by:   oesteban
# @Last Modified time: 2016-05-06 12:11:18
""" A QC workflow for anatomical MRI """
import os
import os.path as op
from nipype.pipeline import engine as pe
from nipype.interfaces import io as nio
from nipype.interfaces import utility as niu
from nipype.interfaces import fsl
from nipype.interfaces import ants
from nipype.interfaces.afni import preprocess as afp

from mriqc.workflows.utils import fwhm_dict
from mriqc.interfaces.qc import StructuralQC
from mriqc.interfaces.anatomical import ArtifactMask
from mriqc.interfaces.viz import PlotMosaic
from mriqc.utils.misc import bids_getfile, bids_path
from mriqc.data.getters import get_mni_template

def anat_qc_workflow(name='MRIQC_Anat', settings=None):
    """
    One-subject-one-session-one-run pipeline to extract the NR-IQMs from
    anatomical images
    """
    if settings is None:
        settings = {}

    workflow = pe.Workflow(name=name)
    deriv_dir = op.abspath('./derivatives')
    if 'work_dir' in settings.keys():
        deriv_dir = op.abspath(op.join(settings['work_dir'], 'derivatives'))

    if not op.exists(deriv_dir):
        os.makedirs(deriv_dir)
    # Define workflow, inputs and outputs
    inputnode = pe.Node(niu.IdentityInterface(
        fields=['bids_root', 'subject_id', 'session_id',
                'run_id']), name='inputnode')
    outputnode = pe.Node(niu.IdentityInterface(fields=['out_json']), name='outputnode')


    # 0. Get data
    datasource = pe.Node(niu.Function(
        input_names=['bids_root', 'data_type', 'subject_id', 'session_id', 'run_id'],
        output_names=['anatomical_scan'], function=bids_getfile), name='datasource')
    datasource.inputs.data_type = 'anat'


    # 1a. Reorient anatomical image
    arw = mri_reorient_wf()
    # 1b. Estimate bias
    n4itk = pe.Node(ants.N4BiasFieldCorrection(dimension=3, save_bias=True), name='Bias')
    # 2. Skull-stripping (afni)
    asw = skullstrip_wf()
    mask = pe.Node(fsl.ApplyMask(), name='MaskAnatomical')
    # 3. Head mask (including nasial-cerebelum mask)
    hmsk = headmsk_wf()
    # 4. Air mask (with and without artifacts)
    amw = airmsk_wf(save_memory=settings.get('save_memory', False),
                    ants_settings=settings.get('ants_settings', None))

    # Brain tissue segmentation
    segment = pe.Node(fsl.FAST(
        img_type=1, segments=True, out_basename='segment'), name='segmentation')

    # AFNI check smoothing
    fwhm = pe.Node(afp.FWHMx(combine=True, detrend=True), name='smoothness')
    # fwhm.inputs.acf = True  # add when AFNI >= 16

    # Compute python-coded measures
    measures = pe.Node(StructuralQC(), 'measures')

    # Plot mosaic
    plot = pe.Node(PlotMosaic(), name='plot_mosaic')
    merg = pe.Node(niu.Merge(3), name='plot_metadata')

    # Connect all nodes
    workflow.connect([
        (inputnode, datasource, [('bids_root', 'bids_root'),
                                 ('subject_id', 'subject_id'),
                                 ('session_id', 'session_id'),
                                 ('run_id', 'run_id')]),
        (datasource, arw, [('anatomical_scan', 'inputnode.in_file')]),
        (arw, asw, [('outputnode.out_file', 'inputnode.in_file')]),
        (arw, n4itk, [('outputnode.out_file', 'input_image')]),
        # (asw, n4itk, [('outputnode.out_mask', 'mask_image')]),
        (n4itk, mask, [('output_image', 'in_file')]),
        (asw, mask, [('outputnode.out_mask', 'mask_file')]),
        (mask, segment, [('out_file', 'in_files')]),
        (n4itk, hmsk, [('output_image', 'inputnode.in_file')]),
        (segment, hmsk, [('tissue_class_map', 'inputnode.in_segm')]),
        (n4itk, measures, [('output_image', 'in_noinu')]),
        (arw, measures, [('outputnode.out_file', 'in_file')]),
        (arw, fwhm, [('outputnode.out_file', 'in_file')]),
        (asw, fwhm, [('outputnode.out_mask', 'mask')]),

        (arw, amw, [('outputnode.out_file', 'inputnode.in_file')]),
        (n4itk, amw, [('output_image', 'inputnode.in_noinu')]),
        (asw, amw, [('outputnode.out_mask', 'inputnode.in_mask')]),
        (hmsk, amw, [('outputnode.out_file', 'inputnode.head_mask')]),

        (amw, measures, [('outputnode.out_file', 'air_msk')]),
        (amw, measures, [('outputnode.artifact_msk', 'artifact_msk')]),

        (segment, measures, [('tissue_class_map', 'in_segm'),
                             ('partial_volume_files', 'in_pvms')]),
        (n4itk, measures, [('bias_image', 'in_bias')]),
        (arw, plot, [('outputnode.out_file', 'in_file')]),
        (inputnode, plot, [('subject_id', 'subject')]),
        (inputnode, merg, [('session_id', 'in1'),
                           ('run_id', 'in2')]),
        (merg, plot, [('out', 'metadata')])
    ])

    if settings.get('mask_mosaic', False):
        workflow.connect(asw, 'outputnode.out_file', plot, 'in_mask')

    # Save mosaic to well-formed path
    mvplot = pe.Node(niu.Rename(
        format_string='anatomical_%(subject_id)s_%(session_id)s_%(run_id)s',
        keep_ext=True), name='rename_plot')
    dsplot = pe.Node(nio.DataSink(
        base_directory=settings['work_dir'], parameterization=False), name='ds_plot')
    workflow.connect([
        (inputnode, mvplot, [('subject_id', 'subject_id'),
                             ('session_id', 'session_id'),
                             ('run_id', 'run_id')]),
        (plot, mvplot, [('out_file', 'in_file')]),
        (mvplot, dsplot, [('out_file', '@mosaic')])
    ])

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
                               ('run_id', 'run_id')]),
        (plot, datasink, [('out_file', 'mosaic_file')]),
        (fwhm, datasink, [(('fwhm', fwhm_dict), 'fwhm')]),
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
                              ('qi2', 'qi2'),
                              ('cjv', 'cjv')]),
        (out_name, datasink, [('out_file', 'out_file')]),
        (datasink, outputnode, [('out_file', 'out_file')])
    ])
    return workflow


def mri_reorient_wf(name='ReorientWorkflow'):
    """A workflow to reorient images to 'RPI' orientation"""
    workflow = pe.Workflow(name=name)

    inputnode = pe.Node(niu.IdentityInterface(fields=['in_file']),
                        name='inputnode')
    outputnode = pe.Node(niu.IdentityInterface(
        fields=['out_file']), name='outputnode')

    deoblique = pe.Node(afp.Refit(deoblique=True), name='deoblique')
    reorient = pe.Node(afp.Resample(
        orientation='RPI', outputtype='NIFTI_GZ'), name='reorient')
    workflow.connect([
        (inputnode, deoblique, [('in_file', 'in_file')]),
        (deoblique, reorient, [('out_file', 'in_file')]),
        (reorient, outputnode, [('out_file', 'out_file')])
    ])
    return workflow


def headmsk_wf(name='HeadMaskWorkflow'):
    """Computes a head mask as in [Mortamet2009]_."""

    has_dipy = False
    try:
        from dipy.denoise import nlmeans
        from nipype.interfaces.dipy import Denoise
        has_dipy = True
    except ImportError:
        pass

    workflow = pe.Workflow(name=name)
    inputnode = pe.Node(niu.IdentityInterface(fields=['in_file', 'in_segm']),
                        name='inputnode')
    outputnode = pe.Node(niu.IdentityInterface(fields=['out_file']), name='outputnode')

    if not has_dipy:
        # Alternative for when dipy is not installed
        bet = pe.Node(fsl.BET(surfaces=True), name='fsl_bet')
        workflow.connect([
            (inputnode, bet, [('in_file', 'in_file')]),
            (bet, outputnode, [('outskin_mask_file', 'out_file')])
        ])
        return workflow

    getwm = pe.Node(niu.Function(
        input_names=['in_file'], output_names=['out_file'], function=_get_wm), name='GetWM')
    denoise = pe.Node(Denoise(), name='Denoise')
    gradient = pe.Node(niu.Function(
        input_names=['in_file'], output_names=['out_file'], function=image_gradient), name='Grad')
    thresh = pe.Node(niu.Function(
        input_names=['in_file'], output_names=['out_file'], function=gradient_threshold),
                     name='GradientThreshold')

    workflow.connect([
        (inputnode, getwm, [('in_segm', 'in_file')]),
        (inputnode, denoise, [('in_file', 'in_file')]),
        (getwm, denoise, [('out_file', 'noise_mask')]),
        (denoise, gradient, [('out_file', 'in_file')]),
        (gradient, thresh, [('out_file', 'in_file')]),
        (thresh, outputnode, [('out_file', 'out_file')])
    ])

    return workflow


def airmsk_wf(name='AirMaskWorkflow', save_memory=False, ants_settings=None):
    """Implements the Step 1 of [Mortamet2009]_."""
    import pkg_resources as pkgr
    workflow = pe.Workflow(name=name)

    inputnode = pe.Node(niu.IdentityInterface(
        fields=['in_file', 'in_noinu', 'in_mask', 'head_mask']), name='inputnode')
    outputnode = pe.Node(niu.IdentityInterface(fields=['out_file', 'artifact_msk']),
                         name='outputnode')

    antsparms = pe.Node(nio.JSONFileGrabber(), name='ants_settings')
    antsparms.inputs.in_file = (
        ants_settings if ants_settings is not None else pkgr.resource_filename(
            'mriqc', 'data/ants_settings.json'))

    def _invt_flags(transforms):
        return [True] * len(transforms)

    # Spatial normalization, using ANTs
    norm = pe.Node(ants.Registration(dimension=3), name='normalize')

    if save_memory:
        norm.inputs.fixed_image = op.join(get_mni_template(), 'MNI152_T1_2mm.nii.gz')
        norm.inputs.fixed_image_mask = op.join(get_mni_template(),
                                               'MNI152_T1_2mm_brain_mask.nii.gz')
    else:
        norm.inputs.fixed_image = op.join(get_mni_template(), 'MNI152_T1_1mm.nii.gz')
        norm.inputs.fixed_image_mask = op.join(get_mni_template(),
                                               'MNI152_T1_1mm_brain_mask.nii.gz')

    invt = pe.Node(ants.ApplyTransforms(
        dimension=3, default_value=1, interpolation='NearestNeighbor'), name='invert_xfm')
    invt.inputs.input_image = op.join(get_mni_template(), 'MNI152_T1_1mm_brain_bottom.nii.gz')

    # Combine and invert mask
    combine = pe.Node(niu.Function(
        input_names=['head_mask', 'artifact_msk'], output_names=['out_file'],
        function=combine_masks), name='combine_masks')

    qi1 = pe.Node(ArtifactMask(), name='ArtifactMask')

    workflow.connect([
        (antsparms, norm, [
            ('initial_moving_transform_com', 'initial_moving_transform_com'),
            ('winsorize_lower_quantile', 'winsorize_lower_quantile'),
            ('winsorize_upper_quantile', 'winsorize_upper_quantile'),
            ('float', 'float'),
            ('transforms', 'transforms'),
            ('transform_parameters', 'transform_parameters'),
            ('number_of_iterations', 'number_of_iterations'),
            ('convergence_window_size', 'convergence_window_size'),
            ('metric', 'metric'),
            ('metric_weight', 'metric_weight'),
            ('radius_or_number_of_bins', 'radius_or_number_of_bins'),
            ('sampling_strategy', 'sampling_strategy'),
            ('sampling_percentage', 'sampling_percentage'),
            ('smoothing_sigmas', 'smoothing_sigmas'),
            ('shrink_factors', 'shrink_factors'),
            ('convergence_threshold', 'convergence_threshold'),
            ('sigma_units', 'sigma_units'),
            ('use_estimate_learning_rate_once', 'use_estimate_learning_rate_once'),
            ('use_histogram_matching', 'use_histogram_matching')
        ]),
        (inputnode, qi1, [('in_file', 'in_file')]),
        (inputnode, norm, [('in_noinu', 'moving_image'),
                           ('in_mask', 'moving_image_mask')]),
        (norm, invt, [('forward_transforms', 'transforms'),
                      (('forward_transforms', _invt_flags), 'invert_transform_flags')]),
        (inputnode, invt, [('in_mask', 'reference_image')]),
        (inputnode, combine, [('head_mask', 'head_mask')]),
        (invt, combine, [('output_image', 'artifact_msk')]),
        (combine, qi1, [('out_file', 'air_msk')]),
        (qi1, outputnode, [('out_air_msk', 'out_file'),
                           ('out_art_msk', 'artifact_msk')])

    ])
    return workflow


def skullstrip_wf(name='SkullStripWorkflow'):
    """ Skull-stripping workflow """

    workflow = pe.Workflow(name=name)
    inputnode = pe.Node(niu.IdentityInterface(fields=['in_file']),
                        name='inputnode')
    outputnode = pe.Node(niu.IdentityInterface(fields=['out_file', 'out_mask', 'head_mask']),
                         name='outputnode')

    sstrip = pe.Node(afp.SkullStrip(outputtype='NIFTI_GZ'), name='skullstrip')
    sstrip_orig_vol = pe.Node(afp.Calc(
        expr='a*step(b)', outputtype='NIFTI_GZ'), name='sstrip_orig_vol')
    binarize = pe.Node(fsl.Threshold(args='-bin', thresh=1.e-3), name='binarize')

    workflow.connect([
        (inputnode, sstrip, [('in_file', 'in_file')]),
        (inputnode, sstrip_orig_vol, [('in_file', 'in_file_a')]),
        (sstrip, sstrip_orig_vol, [('out_file', 'in_file_b')]),
        (sstrip_orig_vol, binarize, [('out_file', 'in_file')]),
        (sstrip_orig_vol, outputnode, [('out_file', 'out_file')]),
        (binarize, outputnode, [('out_file', 'out_mask')])
    ])
    return workflow

def _get_wm(in_file, wm_val=3, out_file=None):
    import os.path as op
    import numpy as np
    import nibabel as nb
    from scipy.ndimage import gaussian_gradient_magnitude as gradient

    if out_file is None:
        fname, ext = op.splitext(op.basename(in_file))
        if ext == '.gz':
            fname, ext2 = op.splitext(fname)
            ext = ext2 + ext
        out_file = op.abspath('%s_wm%s' % (fname, ext))

    imnii = nb.load(in_file)
    data = imnii.get_data().astype(np.uint8)
    msk = np.zeros_like(data)
    msk[data == wm_val] = 1
    nb.Nifti1Image(msk, imnii.get_affine(), imnii.get_header()).to_filename(out_file)
    return out_file

def combine_masks(head_mask, artifact_msk, out_file=None):
    """Computes an air mask from the head and artifact masks"""
    import os.path as op
    import numpy as np
    import nibabel as nb
    from scipy import ndimage as sim

    if out_file is None:
        fname, ext = op.splitext(op.basename(head_mask))
        if ext == '.gz':
            fname, ext2 = op.splitext(fname)
            ext = ext2 + ext
        out_file = op.abspath('%s_combined%s' % (fname, ext))

    imnii = nb.load(head_mask)
    hmdata = imnii.get_data()

    msk = np.ones_like(hmdata, dtype=np.uint8)
    msk[hmdata == 1] = 0

    adata = nb.load(artifact_msk).get_data()
    msk[adata == 1] = 0
    nb.Nifti1Image(msk, imnii.get_affine(), imnii.get_header()).to_filename(out_file)
    return out_file

def image_gradient(in_file, compute_abs=True, out_file=None):
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
        out_file = op.abspath('%s_grad%s' % (fname, ext))

    imnii = nb.load(in_file)
    data = imnii.get_data().astype(np.float32)  # pylint: disable=no-member
    range_max = np.percentile(data.reshape(-1), 90.)
    data *= (100/range_max)
    sigma = 1e-4 * data[data > 0].std(ddof=1)  # pylint: disable=no-member
    grad = gradient(data, sigma)

    while grad.sum() < 1.e4:
        sigma *= 1.5
        grad = gradient(data, sigma)

    if compute_abs:
        grad = np.abs(grad)

    nb.Nifti1Image(grad, imnii.get_affine(), imnii.get_header()).to_filename(out_file)
    return out_file

def gradient_threshold(in_file, thresh=1.0, out_file=None):
    """ Compute a threshold from the histogram of the magnitude gradient image """
    import os.path as op
    import numpy as np
    import nibabel as nb
    from scipy import ndimage as sim

    thresh *= 1e-2
    if out_file is None:
        fname, ext = op.splitext(op.basename(in_file))
        if ext == '.gz':
            fname, ext2 = op.splitext(fname)
            ext = ext2 + ext
        out_file = op.abspath('%s_gradmask%s' % (fname, ext))


    imnii = nb.load(in_file)
    data = imnii.get_data()
    hist, bin_edges = np.histogram(data[data > 0], bins=128, density=True)  # pylint: disable=no-member

    # Find threshold at 1% frequency
    for i, freq in reversed(list(enumerate(hist))):
        binw = bin_edges[i+1] - bin_edges[i]
        if (freq * binw) >= thresh:
            out_thresh = 0.5 * binw
            break

    mask = np.zeros_like(data, dtype=np.uint8)  # pylint: disable=no-member
    mask[data > out_thresh] = 1
    struc = sim.iterate_structure(sim.generate_binary_structure(3, 2), 2)
    mask = sim.binary_opening(mask, struc).astype(np.uint8)  # pylint: disable=no-member

    # Remove small objects
    label_im, nb_labels = sim.label(mask)
    if nb_labels > 2:
        sizes = sim.sum(mask, label_im, range(nb_labels + 1))
        ordered = list(reversed(sorted(zip(sizes, range(nb_labels + 1)))))
        for _, label in ordered[2:]:
            mask[label_im == label] = 0

    mask = sim.binary_closing(mask, struc).astype(np.uint8)  # pylint: disable=no-member
    mask = sim.binary_fill_holes(mask, struc).astype(np.uint8)  # pylint: disable=no-member

    hdr = imnii.get_header().copy()
    hdr.set_data_dtype(np.uint8)  # pylint: disable=no-member
    nb.Nifti1Image(mask, imnii.get_affine(), hdr).to_filename(out_file)
    return out_file
