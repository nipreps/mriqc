#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# @Author: oesteban
# @Date:   2016-01-05 11:24:05
# @Email:  code@oscaresteban.es
# @Last modified by:   oesteban
# @Last Modified time: 2016-02-23 12:05:06
""" A QC workflow for anatomical MRI """
import os.path as op
from nipype.pipeline import engine as pe
from nipype.algorithms import misc as nam
from nipype.interfaces import io as nio
from nipype.interfaces import utility as niu
from nipype.interfaces import fsl
from nipype.interfaces import ants
from nipype.interfaces.afni import preprocess as afp

from ..interfaces.qc import StructuralQC
from ..interfaces.viz import Report
from ..utils.misc import reorder_csv


def anat_qc_workflow(name='aMRIQC', settings=None, sub_list=None):
    """ The anatomical quality control workflow """
    from ..interfaces.viz import PlotMosaic

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
        input_names=['in_qc', 'subject_id', 'session_id', 'scan_id', 'fwhm',],
        output_names=['out_qc'], function=_merge_dicts), name='merge_qc')
    # Import measures from QAP
    # measures = pe.Node(niu.Function(
    #     input_names=['anatomical_reorient', 'head_mask_path',
    #                  'anatomical_segs', 'bias_image', 'fwhm',
    #                  'subject_id', 'session_id', 'scan_id'],
    #     output_names=['out_qc'], function=qc_anat), name='measures')

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
        (datasource, mergqc, [('subject_id', 'subject_id'),
                              ('session_id', 'session_id'),
                              ('scan_id', 'scan_id')]),
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
    from nipype.interfaces.fsl.maths import MathsCommand
    from qap.workflows.utils import select_thresh, slice_head_mask

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

    slice_msk = pe.Node(niu.Function(
        input_names=['infile', 'transform', 'standard'],
        output_names=['outfile_path'], function=slice_head_mask), name='slice_msk')

    combine = pe.Node(fsl.BinaryMaths(
        operation='add', args='-bin'), name='qap_headmask_combine_masks')

    # Get linear mapping to normalized (template) space
    flirt = pe.Node(fsl.FLIRT(cost='corratio'), name='spatial_normalization')

    workflow.connect([
        (inputnode, slice_msk, [(('in_template', _default_template), 'standard')]),
        (inputnode, maskav, [('in_file', 'in_file')]),
        (maskav, hist, [(('out_file', _post_maskav), 'max_value')]),
        (inputnode, hist, [('in_file', 'in_file')]),
        (inputnode, binarize, [('in_file', 'in_file')]),
        (inputnode, slice_msk, [('in_file', 'infile')]),
        (inputnode, flirt, [
            ('in_brain', 'in_file'),
            (('in_template', _default_template), 'reference')]),
        (flirt, slice_msk, [('out_matrix_file', 'transform')]),
        (hist, binarize, [(('out_show', _post_hist), 'thresh')]),
        (binarize, dilate, [('out_file', 'in_file')]),
        (dilate, erode, [('out_file', 'in_file')]),
        (erode, combine, [('out_file', 'in_file')]),
        (slice_msk, combine, [('outfile_path', 'operand_file')]),
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


def _merge_dicts(in_qc, subject_id, session_id, scan_id, fwhm,
                 site_name=None):
    in_qc['subject_id'] = subject_id
    in_qc['session_id'] = session_id
    in_qc['scan_id'] = scan_id

    if site_name is not None:
        in_qc['site_name'] = site_name

    in_qc.update({'fwhm_x': fwhm[0], 'fwhm_y': fwhm[1], 'fwhm_z': fwhm[2],
                  'fwhm': fwhm[3]})

    return in_qc

def qc_anat(anatomical_reorient, head_mask_path, anatomical_segs, bias_image, fwhm,
            subject_id, session_id, scan_id, site_name=None, out_vox=True):
    """
    A wrapper for the QC measures calculation imported from QAP.
    Also adds some measures regarding the inhomogeneity field.
    """
    import nibabel as nb
    import numpy as np
    from nipype import logging
    from qap.spatial_qc import summary_mask, snr, cnr, fber, efc, artifacts
    from qap.qap_utils import load_image, load_mask

    iflogger = logging.getLogger('interface')

    # Load the data
    im_anat = nb.load(anatomical_reorient)
    fg_mask = load_mask(head_mask_path, anatomical_reorient)
    bg_mask = 1 - fg_mask
    gm_mask = load_mask(anatomical_segs[1], anatomical_reorient)
    wm_mask = load_mask(anatomical_segs[2], anatomical_reorient)
    csf_mask = load_mask(anatomical_segs[0], anatomical_reorient)

    out_qc = {'subject': subject_id, 'session': session_id, 'scan': scan_id}
    if site_name is not None:
        out_qc['site_name'] = site_name

    out_qc.update({'fwhm_x': fwhm[0], 'fwhm_y': fwhm[1], 'fwhm_z': fwhm[2],
                   'fwhm': fwhm[3]})

    out_qc.update({'size_x': im_anat.shape[0],
                   'size_y': im_anat.shape[1],
                   'size_z': im_anat.shape[2]})
    out_qc.update({'spacing_%s' % i: v
                   for i, v in zip(['x', 'y', 'z'], im_anat.get_header().get_zooms()[:3])})

    try:
        out_qc.update({'size_t': im_anat.shape[3]})
    except IndexError:
        pass

    try:
        out_qc.update({'tr': im_anat.get_header().get_zooms()[3]})
    except IndexError:
        pass


    im_anat = load_image(anatomical_reorient)

    # Summary Measures
    out_qc['fg_mean'], out_qc['fg_std'], out_qc[
        'fg_size'] = summary_mask(im_anat, fg_mask)
    out_qc['bg_mean'], out_qc['bg_std'], out_qc[
        'bg_size'] = summary_mask(im_anat, bg_mask)

    # More Summary Measures
    out_qc['gm_mean'], out_qc['gm_std'], out_qc[
        'gm_size'] = summary_mask(im_anat, gm_mask)
    out_qc['wm_mean'], out_qc['wm_std'], out_qc[
        'wm_size'] = summary_mask(im_anat, wm_mask)
    out_qc['csf_mean'], out_qc['csf_std'], out_qc[
        'csf_size'] = summary_mask(im_anat, csf_mask)

    # FBER, EFC, artifact
    out_qc.update({'fber': fber(im_anat, fg_mask),
                   'efc': efc(im_anat),
                   'qi1': artifacts(im_anat, fg_mask, calculate_qi2=False)[0],
                   'snr': snr(out_qc['fg_mean'], out_qc['bg_std']),
                   'cnr': cnr(out_qc['gm_mean'], out_qc['wm_mean'], out_qc['bg_std'])})
    iflogger.info('QC measures: %s', out_qc)
    return out_qc
