#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# @Author: oesteban
# @Date:   2016-01-05 11:24:05
# @Email:  code@oscaresteban.es
# @Last modified by:   oesteban
# @Last Modified time: 2016-03-25 16:38:27
""" A QC workflow for anatomical MRI """
import os
import os.path as op
from nipype.pipeline import engine as pe
from nipype.algorithms import misc as nam
from nipype.interfaces import io as nio
from nipype.interfaces import utility as niu
from nipype.interfaces import fsl
from nipype.interfaces import ants
from nipype.interfaces.afni import preprocess as afp

from ..interfaces.qc import StructuralQC
from ..interfaces.viz import Report, PlotMosaic
from ..utils.misc import reorder_csv, rotate_files


def anat_qc_workflow(name='aMRIQC', settings=None, sub_list=None):
    """ The anatomical quality control workflow """

    if settings is None:
        settings = {}

    if sub_list is None:
        sub_list = []

    # Define workflow, inputs and outputs
    workflow = pe.Workflow(name=name)

    inputnode = pe.Node(niu.IdentityInterface(fields=['data']),
                        name='inputnode')
    datasource = pe.Node(niu.IdentityInterface(
        fields=['anatomical_scan', 'subject_id', 'session_id', 'scan_id',
                'site_name']), name='datasource')

    if sub_list:
        inputnode.iterables = [('data', [list(s) for s in sub_list])]

        dsplit = pe.Node(niu.Split(splits=[1, 1, 1, 1], squeeze=True),
                         name='datasplit')
        workflow.connect([
            (inputnode, dsplit, [('data', 'inlist')]),
            (dsplit, datasource, [('out1', 'subject_id'),
                                  ('out2', 'session_id'),
                                  ('out3', 'scan_id'),
                                  ('out4', 'anatomical_scan')])
        ])

    outputnode = pe.Node(niu.IdentityInterface(
        fields=['qc', 'mosaic', 'out_csv', 'out_group']), name='outputnode')

    measures = pe.Node(StructuralQC(), 'measures')
    mergqc = pe.Node(niu.Function(
        input_names=['in_qc', 'subject_id', 'metadata', 'fwhm'],
        output_names=['out_qc'], function=_merge_dicts), name='merge_qc')

    arw = mri_reorient_wf()  # 1. Reorient anatomical image
    n4itk = pe.Node(ants.N4BiasFieldCorrection(dimension=3, save_bias=True,
                    bspline_fitting_distance=30.), name='Bias')
    asw = skullstrip_wf()    # 2. Skull-stripping (afni)
    mask = pe.Node(fsl.ApplyMask(), name='MaskAnatomical')
    amw = airmsk_wf()

    # Brain tissue segmentation
    segment = pe.Node(fsl.FAST(
        img_type=1, segments=True, out_basename='segment'), name='segmentation')

    # AFNI check smoothing
    fwhm = pe.Node(afp.FWHMx(combine=True, detrend=True), name='smoothness')
    # fwhm.inputs.acf = True  # add when AFNI >= 16

    # Plot mosaic
    plot = pe.Node(PlotMosaic(), name='plot_mosaic')
    merg = pe.Node(niu.Merge(3), name='plot_metadata')

    workflow.connect([
        (datasource, arw, [('anatomical_scan', 'inputnode.in_file')]),
        (arw, asw, [('outputnode.out_file', 'inputnode.in_file')]),
        (arw, n4itk, [('outputnode.out_file', 'input_image')]),
        (asw, n4itk, [('outputnode.out_mask', 'weight_image')]),
        (n4itk, mask, [('output_image', 'in_file')]),
        (asw, mask, [('outputnode.out_mask', 'mask_file')]),
        (mask, segment, [('out_file', 'in_files')]),
        (arw, measures, [('outputnode.out_file', 'in_file')]),
        (n4itk, fwhm, [('output_image', 'in_file')]),
        (asw, fwhm, [('outputnode.out_mask', 'mask')]),

        (n4itk, amw, [('output_image', 'inputnode.in_file')]),
        (asw, amw, [('outputnode.out_mask', 'inputnode.in_mask')]),
        (asw, amw, [('outputnode.head_mask', 'inputnode.head_mask')]),

        (fwhm, mergqc, [('fwhm', 'fwhm')]),
        (amw, measures, [('outputnode.out_file', 'air_msk')]),
        (segment, measures, [('tissue_class_map', 'in_segm'),
                             ('partial_volume_files', 'in_pvms')]),
        (n4itk, measures, [('bias_image', 'in_bias')]),
        (arw, plot, [('outputnode.out_file', 'in_file')]),
        (datasource, plot, [('subject_id', 'subject')]),
        (datasource, merg, [('session_id', 'in1'),
                            ('scan_id', 'in2'),
                            ('site_name', 'in3')]),
        (datasource, mergqc, [('subject_id', 'subject_id')]),
        (merg, mergqc, [('out', 'metadata')]),
        (merg, plot, [('out', 'metadata')]),
        (measures, mergqc, [('out_qc', 'in_qc')]),
        (mergqc, outputnode, [('out_qc', 'qc')]),
        (plot, outputnode, [('out_file', 'mosaic')]),
    ])

    if settings.get('mask_mosaic', False):
        workflow.connect(asw, 'outputnode.out_file', plot, 'in_mask')

    # Save mosaic to well-formed path
    mvplot = pe.Node(niu.Rename(
        format_string='anatomical_%(subject_id)s_%(session_id)s_%(scan_id)s',
        keep_ext=True), name='rename_plot')
    dsplot = pe.Node(nio.DataSink(
        base_directory=settings['work_dir'], parameterization=False), name='ds_plot')
    workflow.connect([
        (datasource, mvplot, [('subject_id', 'subject_id'),
                              ('session_id', 'session_id'),
                              ('scan_id', 'scan_id')]),
        (plot, mvplot, [('out_file', 'in_file')]),
        (mvplot, dsplot, [('out_file', '@mosaic')])
    ])

    # Export to CSV
    out_csv = op.join(settings['output_dir'], 'aMRIQC.csv')
    rotate_files(out_csv)

    to_csv = pe.Node(nam.AddCSVRow(in_file=out_csv), name='write_csv')
    re_csv0 = pe.JoinNode(niu.Function(input_names=['csv_file'], output_names=['out_file'],
                                       function=reorder_csv), joinsource='inputnode',
                          joinfield='csv_file', name='reorder_anat')
    report0 = pe.Node(
        Report(qctype='anatomical', settings=settings), name='AnatomicalReport')
    if sub_list:
        report0.inputs.sub_list = sub_list

    workflow.connect([
        (mergqc, to_csv, [('out_qc', '_outputs')]),
        (to_csv, re_csv0, [('csv_file', 'csv_file')]),
        (re_csv0, outputnode, [('out_file', 'out_csv')]),
        (re_csv0, report0, [('out_file', 'in_csv')]),
        (report0, outputnode, [('out_group', 'out_group')])
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
    """Implements the Step 0 of [Mortamet2009]_."""

    try:
        from nipype.interfaces.dipy import Denoise
        has_dipy = True
    except ImportError:
        has_dipy = False

    workflow = pe.Workflow(name=name)
    inputnode = pe.Node(niu.IdentityInterface(fields=['in_file', 'wm_mask']),
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

    denoise = pe.Node(Denoise(), name='Denoise')
    gradient = pe.Node(niu.Function(
        input_names=['in_file'], output_names=['out_file'], function=image_gradient), name='Grad')
    thresh = pe

    workflow.connect([
        (inputnode, denoise, [('in_file', 'in_file'),
                              ('wm_mask', 'noise_mask')]),
        (denoise, gradient, [('out_file', 'in_file')]),

    ])

    return workflow


def airmsk_wf(name='AirMaskWorkflow'):
    """Implements the Step 1 of [Mortamet2009]_."""
    import pkg_resources as p
    workflow = pe.Workflow(name=name)

    inputnode = pe.Node(niu.IdentityInterface(
        fields=['in_file', 'in_mask', 'head_mask']), name='inputnode')
    outputnode = pe.Node(niu.IdentityInterface(fields=['out_file']), name='outputnode')

    # Get linear mapping to normalized (template) space
    flirt = pe.Node(fsl.FLIRT(cost='corratio', dof=12, bgvalue=0), name='spatial_normalization')
    flirt.inputs.reference = p.resource_filename('mriqc', 'data/MNI152_T1_1mm_brain.nii.gz')
    flirt.inputs.ref_weight = p.resource_filename('mriqc', 'data/MNI152_T1_1mm_brain_mask.nii.gz')

    # Invert affine matrix
    invt = pe.Node(fsl.ConvertXFM(invert_xfm=True), name='invert_xfm')

    # Normalize the bottom part of the mask to the image space
    mask = pe.Node(fsl.ApplyXfm(bgvalue=1, apply_xfm=True, interp='nearestneighbour'),
                   name='ApplyXfmToMask')
    mask.inputs.in_file = p.resource_filename('mriqc', 'data/MNI152_T1_1mm_brain_bottom.nii.gz')

    # Combine and invert mask
    combine = pe.Node(fsl.BinaryMaths(operation='add', args='-bin'), name='combine_masks')
    invertmsk = pe.Node(fsl.BinaryMaths(operation='mul', operand_value=-1.0, args='-add 1'),
                        name='InvertMask')

    workflow.connect([
        (inputnode, flirt, [('in_file', 'in_file'),
                            ('in_mask', 'in_weight')]),
        (flirt, invt, [('out_matrix_file', 'in_file')]),
        (invt, mask, [('out_file', 'in_matrix_file')]),
        (inputnode, mask, [('in_mask', 'reference')]),
        (inputnode, combine, [('head_mask', 'in_file')]),
        (mask, combine, [('out_file', 'operand_file')]),
        (combine, invertmsk, [('out_file', 'in_file')]),
        (invertmsk, outputnode, [('out_file', 'out_file')])
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


def _merge_dicts(in_qc, subject_id, metadata, fwhm):
    in_qc['subject'] = subject_id
    in_qc['session'] = metadata[0]
    in_qc['scan'] = metadata[1]

    try:
        in_qc['site_name'] = metadata[2]
    except IndexError:
        pass  # No site_name defined

    in_qc.update({'fwhm_x': fwhm[0], 'fwhm_y': fwhm[1], 'fwhm_z': fwhm[2],
                  'fwhm': fwhm[3]})

    in_qc['snr'] = in_qc.pop('snr_total')
    try:
        in_qc['tr'] = in_qc['spacing_tr']
    except KeyError:
        pass  # TR is not defined

    return in_qc

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

    im = nb.load(in_file)
    data = im.get_data().astype(np.float32)  # pylint: disable=no-member
    sigma = .01 * (np.percentile(data[data > 0], 75.) - np.percentile(data[data > 0], 25.))  # pylint: disable=no-member
    grad = gradient(data, sigma)

    if compute_abs:
        grad = np.abs(grad)

    nb.Nifti1Image(grad, im.get_affine(), im.get_header()).to_filename(out_file)
    return out_file

def gradient_threshold(in_file, thresh=1.0, out_file=None):
    """ Compute a threshold from the histogram of the magnitude gradient image """
    import numpy as np
    import nibabel as nb
    from scipy.ndimage import (generate_binary_structure, iterate_structure,
                               binary_closing, binary_fill_holes)
    thresh *= 1e-2
    if out_file is None:
        fname, ext = op.splitext(op.basename(in_file))
        if ext == '.gz':
            fname, ext2 = op.splitext(fname)
            ext = ext2 + ext
        out_file = op.abspath('%s_gradmask%s' % (fname, ext))


    im = nb.load(in_file)
    data = im.get_data()
    hist, bin_edges = np.histogram(data[data > 0], bins=128, density=True)  # pylint: disable=no-member

    for i, freq in reversed(list(enumerate(hist))):
        if freq >= thresh:
            out_thresh = 0.5 * (bin_edges[i+1] - bin_edges[i])
            break

    mask = np.zeros_like(data, dtype=np.uint8)  # pylint: disable=no-member
    mask[data > out_thresh] = 1
    struc = iterate_structure(generate_binary_structure(3, 1), 2)
    mask = binary_closing(mask, struc).astype(np.uint8)  # pylint: disable=no-member
    mask = binary_fill_holes(mask, struc).astype(np.uint8)  # pylint: disable=no-member

    hdr = im.get_header().copy()
    hdr.set_data_dtype(np.uint8)  # pylint: disable=no-member
    nb.Nifti1Image(mask, im.get_affine(), hdr).to_filename(out_file)
    return out_file
