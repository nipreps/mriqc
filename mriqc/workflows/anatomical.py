#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# @Author: oesteban
# @Date:   2016-01-05 11:24:05
# @Email:  code@oscaresteban.es
# @Last modified by:   oesteban
# @Last Modified time: 2016-03-25 15:10:14
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
    n4itk = pe.Node(ants.N4BiasFieldCorrection(dimension=3, bias_image='output_bias.nii.gz'),
                    name='Bias')
    asw = skullstrip_wf()    # 2. Skull-stripping (afni)
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
        (arw, n4itk, [('outputnode.out_file', 'input_image')]),
        (n4itk, asw, [('output_image', 'inputnode.in_file')]),
        (asw, segment, [('outputnode.out_file', 'in_files')]),
        (n4itk, measures, [('output_image', 'in_file')]),
        (n4itk, fwhm, [('output_image', 'in_file')]),
        (asw, fwhm, [('outputnode.out_mask', 'mask')]),

        (n4itk, amw, [('output_image', 'inputnode.in_file')]),
        (asw, amw, [('outputnode.out_mask', 'inputnode.in_mask')]),
        (asw, amw, [('outputnode.out_mask', 'inputnode.head_mask')]),

        (fwhm, mergqc, [('fwhm', 'fwhm')]),
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
    workflow = pe.Workflow(name=name)


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
    flirt.inputs.in_file = p.resource_filename('mriqc', 'data/MNI152_T1_1mm_brain.nii.gz')
    flirt.inputs.in_weight = p.resource_filename('mriqc', 'data/MNI152_T1_1mm_brain_mask.nii.gz')

    # Normalize the bottom part of the mask to the image space
    mask = pe.Node(fsl.ApplyXfm(bgvalue=1, apply_xfm=True, interp='nearestneighbour'),
                   name='ApplyXfmToMask')
    mask.inputs.in_file = p.resource_filename('mriqc', 'data/MNI152_T1_1mm_brain_bottom.nii.gz')

    # Combine and invert mask
    combine = pe.Node(fsl.BinaryMaths(operation='add', args='-bin'), name='combine_masks')
    invertmsk = pe.Node(fsl.BinaryMaths(operation='mul', operand_value=-1.0, args='-add 1'),
                        name='InvertMask')

    workflow.connect([
        (inputnode, flirt, [('in_file', 'reference'),
                            ('in_mask', 'ref_weight')]),
        (flirt, mask, [('out_matrix_file', 'in_matrix_file')]),
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
    outputnode = pe.Node(niu.IdentityInterface(fields=['out_file', 'out_mask']),
                         name='outputnode')

    sstrip = pe.Node(afp.SkullStrip(outputtype='NIFTI_GZ'), name='skullstrip')
    sstrip_orig_vol = pe.Node(afp.Calc(
        expr='a*step(b)', outputtype='NIFTI_GZ'), name='sstrip_orig_vol')
    binarize = pe.Node(fsl.Threshold(args='-bin'), name='binarize')

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


def slice_head_mask(in_file, in_coords, out_file=None):
    import os.path as op
    import numpy as np
    import nibabel as nb

    # get file info
    in_nii = nb.load(in_file)
    in_header = in_nii.get_header()
    in_aff = in_nii.get_affine()
    in_dims = in_header.get_data_shape()

    coords = []
    for vox in np.loadtxt(in_coords):  # pylint: disable=no-member
        vox = [int(v) for v in vox]

        for i in range(0, 3):
            vox[i] = np.clip(vox[i], 1, in_dims[i] - 1)

        coords.append(np.array(vox))

    # get the vectors connecting the points
    uvector = []
    for a_pt, c_pt in zip(coords[0], coords[2]):
        uvector.append(int(a_pt - c_pt))

    vvector = []
    for b_pt, c_pt in zip(coords[1], coords[2]):
        vvector.append(int(b_pt - c_pt))

    # vector cross product
    nvector = np.cross(uvector, vvector)

    # normalize the vector
    nvector = nvector / np.linalg.norm(nvector, 2)
    constant = np.dot(nvector, np.asarray(coords[0]))

    # now determine the z-coordinate for each pair of x,y
    plane_dict = {}

    for yvox in range(0, in_dims[1]):
        for xvox in range(0, in_dims[0]):
            zvox = (constant - (nvector[0] * xvox + nvector[1] * yvox)) / nvector[2]
            zvox = np.floor(zvox)  # pylint: disable=no-member

            if zvox < 1:
                zvox = 1
            elif zvox > in_dims[2]:
                zvox = in_dims[2]

            plane_dict[(xvox, yvox)] = zvox

    # create the mask
    mask_array = np.zeros(in_dims)

    for i in range(0, in_dims[0]):
        for j in range(0, in_dims[1]):
            for k in range(0, in_dims[2]):
                if plane_dict[(i, j)] > k:
                    mask_array[i, j, k] = 1


    if out_file is None:
        fname, ext = op.splitext(op.basename(in_file))
        if ext == '.gz':
            fname, ext2 = op.splitext(fname)
            ext = ext2 + ext
        out_file = op.abspath('%s_slice_mask%s' % (fname, ext))

    nb.Nifti1Image(mask_array, in_aff, in_header).to_filename(out_file)
    return out_file
