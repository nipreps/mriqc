#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# @Author: oesteban
# @Date:   2016-01-05 11:24:05
# @Email:  code@oscaresteban.es
# @Last modified by:   oesteban
# @Last Modified time: 2016-01-18 08:36:22
""" A QC workflow for anatomical MRI """
import os.path as op
from nipype.pipeline import engine as pe
from nipype.algorithms import misc as nam
from nipype.interfaces import io as nio
from nipype.interfaces import utility as niu
from nipype.interfaces import fsl

from ..interfaces.viz import Report
from ..utils import reorder_csv


def anat_qc_workflow(name='aMRIQC', settings={}, sub_list=[]):
    from ..interfaces.viz import PlotMosaic

    # Define workflow, inputs and outputs
    wf = pe.Workflow(name=name)

    if 'work_dir' in settings.keys():
        wf.base_dir = settings['work_dir']

    inputnode = pe.Node(niu.IdentityInterface(fields=['data']),
                        name='inputnode')
    datasource = pe.Node(niu.IdentityInterface(
        fields=['anatomical_scan', 'subject_id', 'session_id', 'scan_id',
                'site_name']), name='datasource')

    if sub_list:
        inputnode.iterables = [('data', [list(s) for s in sub_list])]

        dsplit = pe.Node(niu.Split(splits=[1, 1, 1, 1], squeeze=True),
                         name='datasplit')
        wf.connect([
            (inputnode, dsplit, [('data', 'inlist')]),
            (dsplit, datasource, [('out1', 'subject_id'),
                                  ('out2', 'session_id'),
                                  ('out3', 'scan_id'),
                                  ('out4', 'anatomical_scan')])
        ])

    outputnode = pe.Node(niu.IdentityInterface(
        fields=['qc', 'mosaic', 'out_csv', 'out_group']), name='outputnode')

    # Import measures from QAP
    measures = pe.Node(niu.Function(
        input_names=['anatomical_reorient', 'head_mask_path',
                     'anatomical_segs', 'subject_id', 'session_id',
                     'scan_id'],
        output_names=['qc'], function=qc_anat), name='measures')

    arw = mri_reorient_wf()  # 1. Reorient anatomical image
    asw = skullstrip_wf()    # 2. Skull-stripping (afni)
    qmw = brainmsk_wf()      # 3. Brain mask (template & 2.)

    # Brain tissue segmentation
    segment = pe.Node(fsl.FAST(
        img_type=1, segments=True, probability_maps=True,
        out_basename='segment'), name='segmentation')

    # Plot mosaic
    plot = pe.Node(PlotMosaic(), name='plot_mosaic')
    merg = pe.Node(niu.Merge(3), name='plot_metadata')

    wf.connect([
        (datasource, measures, [('subject_id', 'subject_id'),
                                ('session_id', 'session_id'),
                                ('scan_id', 'scan_id')]),
        (datasource, arw, [('anatomical_scan', 'inputnode.in_file')]),
        (arw, asw, [('outputnode.out_file',
                     'inputnode.in_file')]),
        (arw, qmw, [('outputnode.out_file', 'inputnode.in_file')]),
        (asw, qmw, [('outputnode.out_file', 'inputnode.in_brain')]),
        (asw, segment,  [('outputnode.out_file', 'in_files')]),
        (arw, measures, [('outputnode.out_file', 'anatomical_reorient')]),
        (qmw, measures, [('outputnode.out_mask', 'head_mask_path')]),
        (segment, measures, [('tissue_class_files', 'anatomical_segs')]),
        (arw, plot, [('outputnode.out_file', 'in_file')]),
        (datasource, plot, [('subject_id', 'subject')]),
        (datasource, merg, [('session_id', 'in1'),
                            ('scan_id', 'in2'),
                            ('site_name', 'in3')]),
        (merg, plot, [('out', 'metadata')]),
        (measures, outputnode, [('qc', 'qc')]),
        (plot, outputnode, [('out_file', 'mosaic')]),
    ])

    if settings.get('mask_mosaic', False):
        wf.connect(qmw, 'outputnode.out_file', plot, 'in_mask')

    # Save mosaic to well-formed path
    mvplot = pe.Node(niu.Rename(
        format_string='anatomical_%(subject_id)s_%(session_id)s_%(scan_id)s',
        keep_ext=True), name='rename_plot')
    dsplot = pe.Node(nio.DataSink(
        base_directory=settings['work_dir'], parameterization=False), name='ds_plot')
    wf.connect([
        (datasource, mvplot, [('subject_id', 'subject_id'),
                              ('session_id', 'session_id'),
                              ('scan_id', 'scan_id')]),
        (plot, mvplot, [('out_file', 'in_file')]),
        (mvplot, dsplot, [('out_file', '@mosaic')])
    ])


    # Export to CSV
    out_csv = op.join(settings['output_dir'], 'aMRIQC.csv')
    to_csv = pe.Node(nam.AddCSVRow(in_file=out_csv), name='write_csv')
    re_csv0 = pe.JoinNode(niu.Function(
        input_names=['csv_file'], output_names=['out_file'], function=reorder_csv),
                          joinsource='inputnode', joinfield='csv_file', name='reorder_anat')
    report0 = pe.Node(Report(qctype='anatomical', settings=settings), name='AnatomicalReport')
    if sub_list:
        report0.inputs.sub_list = sub_list

    wf.connect([
        (measures, to_csv, [('qc', '_outputs')]),
        (to_csv, re_csv0,  [('csv_file', 'csv_file')]),
        (re_csv0, outputnode, [('out_file', 'out_csv')]),
        (re_csv0, report0,  [('out_file', 'in_csv')]),
        (report0, outputnode, [('out_group', 'out_group')])
    ])

    return wf


def mri_reorient_wf(name='ReorientWorkflow'):
    """A workflow to reorient images to 'RPI' orientation"""

    from nipype.interfaces.afni import preprocess as afp

    wf = pe.Workflow(name=name)

    inputnode = pe.Node(niu.IdentityInterface(fields=['in_file']),
                        name='inputnode')
    outputnode = pe.Node(niu.IdentityInterface(
        fields=['out_file']), name='outputnode')

    deoblique = pe.Node(afp.Refit(deoblique=True), name='deoblique')
    reorient = pe.Node(afp.Resample(
        orientation='RPI', outputtype='NIFTI_GZ'), name='reorient')
    wf.connect([
        (inputnode, deoblique, [('in_file', 'in_file')]),
        (deoblique, reorient,  [('out_file', 'in_file')]),
        (reorient, outputnode, [('out_file', 'out_file')])
    ])
    return wf


def brainmsk_wf(name='BrainMaskWorkflow', config={}):
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

    wf = pe.Workflow(name=name)
    inputnode = pe.Node(niu.IdentityInterface(
        fields=['in_file', 'in_brain', 'in_template']), name='inputnode')
    outputnode = pe.Node(niu.IdentityInterface(
        fields=['out_mask', 'out_matrix_file']), name='outputnode')

    sel_th = pe.Node(niu.Function(
        function=select_thresh, input_names=['input_skull'],
        output_names=['thresh']), name='thresh', iterfield=['input_skull'])

    binarize = pe.Node(fsl.Threshold(args='-bin'), name='binarize')
    dilate = pe.Node(MathsCommand(args=' '.join(['-dilM']*6)), name='dilate')
    erode = pe.Node(MathsCommand(args=' '.join(['-eroF']*6)), name='erode')

    slice_msk = pe.Node(niu.Function(
        input_names=['infile', 'transform', 'standard'],
        output_names=['outfile_path'], function=slice_head_mask),
        name='slice_msk')

    combine = pe.Node(fsl.BinaryMaths(
        operation='add', args='-bin'), name='qap_headmask_combine_masks')

    # Get linear mapping to normalized (template) space
    flirt = pe.Node(fsl.FLIRT(cost='corratio'), name='spatial_normalization')

    wf.connect([
        (inputnode, slice_msk,
            [(('in_template', _default_template), 'standard')]),
        (inputnode, sel_th,    [('in_file', 'input_skull')]),
        (inputnode, binarize,  [('in_file', 'in_file')]),
        (inputnode, slice_msk, [('in_file', 'infile')]),
        (inputnode, flirt,     [
            ('in_brain', 'in_file'),
            (('in_template', _default_template), 'reference')]),
        (flirt, slice_msk,     [('out_matrix_file', 'transform')]),
        (sel_th, binarize,     [('thresh', 'thresh')]),
        (binarize, dilate,     [('out_file', 'in_file')]),
        (dilate, erode,        [('out_file', 'in_file')]),
        (erode, combine,       [('out_file', 'in_file')]),
        (slice_msk, combine,   [('outfile_path', 'operand_file')]),
        (combine, outputnode,  [('out_file', 'out_mask')]),
        (flirt, outputnode,    [('out_matrix_file', 'out_matrix_file')]),
    ])
    return wf


def skullstrip_wf(name='SkullStripWorkflow'):
    from nipype.interfaces.afni import preprocess as afp

    wf = pe.Workflow(name=name)
    inputnode = pe.Node(niu.IdentityInterface(fields=['in_file']),
                        name='inputnode')
    outputnode = pe.Node(niu.IdentityInterface(fields=['out_file']),
                         name='outputnode')

    sstrip = pe.Node(afp.SkullStrip(outputtype='NIFTI_GZ'), name='skullstrip')
    sstrip_orig_vol = pe.Node(afp.Calc(
        expr='a*step(b)', outputtype='NIFTI_GZ'), name='sstrip_orig_vol')

    wf.connect([
        (inputnode, sstrip,           [('in_file', 'in_file')]),
        (inputnode, sstrip_orig_vol,  [('in_file', 'in_file_a')]),
        (sstrip, sstrip_orig_vol,     [('out_file', 'in_file_b')]),
        (sstrip_orig_vol, outputnode, [('out_file', 'out_file')])
    ])
    return wf

def qc_anat(anatomical_reorient, head_mask_path,
            anatomical_segs, subject_id, session_id,
            scan_id, site_name=None, out_vox=True):
    from qap.workflows.utils import qap_anatomical_spatial
    return qap_anatomical_spatial(
        anatomical_reorient, head_mask_path, anatomical_segs[1],
        anatomical_segs[2], anatomical_segs[0], subject_id, session_id,
        scan_id, site_name, out_vox)
