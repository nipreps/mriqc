#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# @Author: oesteban
# @Date:   2016-01-05 11:24:05
# @Email:  code@oscaresteban.es
# @Last modified by:   oesteban
# @Last Modified time: 2016-03-04 13:51:59
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


SLICE_MASK_POINTS = [(78., -110., -72.),
                     (-78., -110., -72.),
                     (-1., 91., -29.)]

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
    qmw = brainmsk_wf()      # 3. Brain mask (template & 2.)

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
        (n4itk, qmw, [('output_image', 'inputnode.in_file')]),
        (asw, qmw, [('outputnode.out_file', 'inputnode.in_brain')]),
        (asw, segment, [('outputnode.out_file', 'in_files')]),
        (n4itk, measures, [('output_image', 'in_file')]),
        (n4itk, fwhm, [('output_image', 'in_file')]),
        (qmw, fwhm, [('outputnode.out_mask', 'mask')]),
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
        workflow.connect(qmw, 'outputnode.out_file', plot, 'in_mask')

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


def brainmsk_wf(name='BrainMaskWorkflow'):
    """Computes a brain mask from the original T1 and the skull-stripped"""
    import pkg_resources as p
    from nipype.interfaces.fsl.maths import MathsCommand
    from nipype.interfaces.fsl.utils import WarpPointsFromStd

    def _default_template(in_file):
        from nipype.interfaces.fsl.base import Info
        from os.path import isfile
        from nipype.interfaces.base import isdefined
        if isdefined(in_file) and isfile(in_file):
            return in_file
        return Info.standard_image('MNI152_T1_2mm.nii.gz')

    def _post_maskav(in_file):
        with open(in_file, 'r') as fdesc:
            avg_out = fdesc.readlines()
        avg = int(float(avg_out[-1].split(" ")[0]))
        return int(avg * 3)

    def _post_hist(in_file):
        with open(in_file, 'r') as fdesc:
            hist_out = fdesc.readlines()

        bins = {}
        for line in hist_out:
            if "*" in line and not line.startswith("*"):
                vox_bin = line.replace(" ", "").split(":")[0]
                voxel_value = int(float(vox_bin.split(",")[0]))
                bins[int(vox_bin.split(",")[1])] = voxel_value
        return bins[min(bins.keys())]

    workflow = pe.Workflow(name=name)
    inputnode = pe.Node(niu.IdentityInterface(
        fields=['in_file', 'in_brain', 'in_template']), name='inputnode')
    outputnode = pe.Node(niu.IdentityInterface(
        fields=['out_mask', 'out_matrix_file']), name='outputnode')

    # Compute threshold from histogram and generate mask
    maskav = pe.Node(afp.Maskave(), 'mask_average')
    hist = pe.Node(afp.Hist(nbin=10, showhist=True), 'brain_hist')
    binarize = pe.Node(fsl.Threshold(args='-bin'), name='binarize')
    dilate = pe.Node(MathsCommand(args=' '.join(['-dilM']*6)), name='dilate')
    erode = pe.Node(MathsCommand(args=' '.join(['-eroF']*6)), name='erode')

    msk_coords = pe.Node(WarpPointsFromStd(coord_vox=True), name='msk_coords')
    msk_coords.inputs.in_coords = p.resource_filename('mriqc', 'data/slice_mask_points.txt')

    slice_msk = pe.Node(niu.Function(
        input_names=['in_file', 'in_coords'],
        output_names=['out_file'], function=slice_head_mask), name='slice_msk')

    combine = pe.Node(fsl.BinaryMaths(
        operation='add', args='-bin'), name='headmask_combine_masks')

    # Get linear mapping to normalized (template) space
    flirt = pe.Node(fsl.FLIRT(cost='corratio'), name='spatial_normalization')

    workflow.connect([
        (inputnode, msk_coords, [(('in_template', _default_template), 'std_file')]),
        (inputnode, msk_coords, [('in_file', 'img_file')]),
        (inputnode, slice_msk, [('in_file', 'in_file')]),
        (inputnode, maskav, [('in_file', 'in_file')]),
        (maskav, hist, [(('out_file', _post_maskav), 'max_value')]),
        (inputnode, hist, [('in_file', 'in_file')]),
        (inputnode, binarize, [('in_file', 'in_file')]),
        (inputnode, flirt, [
            ('in_brain', 'in_file'),
            (('in_template', _default_template), 'reference')]),
        (flirt, msk_coords, [('out_matrix_file', 'xfm_file')]),
        (msk_coords, slice_msk, [('out_file', 'in_coords')]),
        (hist, binarize, [(('out_show', _post_hist), 'thresh')]),
        (binarize, dilate, [('out_file', 'in_file')]),
        (dilate, erode, [('out_file', 'in_file')]),
        (erode, combine, [('out_file', 'in_file')]),
        (slice_msk, combine, [('out_file', 'operand_file')]),
        (combine, outputnode, [('out_file', 'out_mask')]),
        (flirt, outputnode, [('out_matrix_file', 'out_matrix_file')]),
    ])
    return workflow


def skullstrip_wf(name='SkullStripWorkflow'):
    """ Skull-stripping workflow """

    workflow = pe.Workflow(name=name)
    inputnode = pe.Node(niu.IdentityInterface(fields=['in_file']),
                        name='inputnode')
    outputnode = pe.Node(niu.IdentityInterface(fields=['out_file']),
                         name='outputnode')

    sstrip = pe.Node(afp.SkullStrip(outputtype='NIFTI_GZ'), name='skullstrip')
    sstrip_orig_vol = pe.Node(afp.Calc(
        expr='a*step(b)', outputtype='NIFTI_GZ'), name='sstrip_orig_vol')

    workflow.connect([
        (inputnode, sstrip, [('in_file', 'in_file')]),
        (inputnode, sstrip_orig_vol, [('in_file', 'in_file_a')]),
        (sstrip, sstrip_orig_vol, [('out_file', 'in_file_b')]),
        (sstrip_orig_vol, outputnode, [('out_file', 'out_file')])
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
