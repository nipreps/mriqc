#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# @Author: oesteban
# @Date:   2016-01-05 16:15:08
# @Email:  code@oscaresteban.es
# @Last modified by:   oesteban
# @Last Modified time: 2016-01-18 16:06:02


import os
import os.path as op
import sys

from nipype.pipeline import engine as pe
from nipype.algorithms import misc as nam
from nipype.interfaces import io as nio
from nipype.interfaces import utility as niu
from nipype.interfaces import fsl
from nipype.interfaces.afni import preprocess as afp

from nipype import logging
logger = logging.getLogger('workflow')


def fmri_qc_workflow(name='fMRIQC', sub_list=[], settings={}):
    from qap.workflows.utils import qap_functional_spatial as qc_fmri_spat
    from qap.workflows.utils import qap_functional_temporal as qc_fmri_temp
    from qap.temporal_qc import fd_jenkinson
    from .utils import fmri_getidx
    from ..interfaces.viz import PlotMosaic, PlotFD

    # Define workflow, inputs and outputs
    workflow = pe.Workflow(name=name)

    if 'work_dir' in settings.keys():
        workflow.base_dir = settings['work_dir']

    inputnode = pe.Node(niu.IdentityInterface(fields=['data']),
                        name='inputnode')
    dsource = pe.Node(niu.IdentityInterface(
        fields=['functional_scan', 'subject_id', 'session_id', 'scan_id',
                'site_name', 'start_idx', 'stop_idx']), name='datasource')
    dsource.inputs.start_idx = 0
    dsource.inputs.stop_idx = None

    get_idx = pe.Node(niu.Function(
        input_names=['in_file', 'start_idx', 'stop_idx'], function=fmri_getidx,
        output_names=['start_idx', 'stop_idx']), name='get_idx')

    if sub_list:
        inputnode.iterables = [('data', [list(s) for s in sub_list])]

        dsplit = pe.Node(niu.Split(splits=[1, 1, 1, 1], squeeze=True),
                         name='datasplit')
        workflow.connect([
            (inputnode, dsplit, [('data', 'inlist')]),
            (dsplit, dsource, [('out1', 'subject_id'),
                                  ('out2', 'session_id'),
                                  ('out3', 'scan_id'),
                                  ('out4', 'functional_scan')])
        ])

    outputnode = pe.Node(niu.IdentityInterface(
        fields=['qc', 'mosaic']), name='outputnode')

    # Measures
    def _empty(val):
        from nipype.interfaces.base import isdefined
        if isdefined(val):
            return val
        return None

    m_spatial = pe.Node(niu.Function(
        input_names=['mean_epi', 'func_brain_mask', 'direction', 'subject_id',
                     'session_id', 'scan_id', 'site_name'],
        output_names=['qc'], function=qc_fmri_spat), name='m_spatial')
    m_spatial.inputs.direction = 'y'  # TODO: handle this parameter

    m_temp = pe.Node(niu.Function(
        input_names=['func_motion_correct', 'func_brain_mask', 'tsnr_volume',
                     'fd_file', 'subject_id', 'session_id', 'scan_id',
                     'site_name'], output_names=['qc'], function=qc_fmri_temp),
                     name='m_temp')

    # Workflow --------------------------------------------------------
    hmcwf = fmri_hmc_workflow(                  # 1. HMC: head motion correct
        st_correct=settings.get('correct_slice_timing', False))
    mean = pe.Node(afp.TStat(                   # 2. Compute mean fmri
        options='-mean', outputtype='NIFTI_GZ'), name='mean')
    bmw = fmri_bmsk_workflow(                   # 3. Compute brain mask
        use_bet=settings.get('use_bet', False))

    fd = pe.Node(niu.Function(
        input_names=['in_file'], output_names=['out_file'],
        function=fd_jenkinson), name='generate_FD_file')
    tsnr = pe.Node(nam.TSNR(), name='compute_tsnr')

    # Merge spatial and temporal measures
    def _merge_dicts(qc_spatial, qc_temporal):
        qc_spatial.update(qc_temporal)
        return qc_spatial

    mqc = pe.Node(niu.Function(
        input_names=['qc_spatial', 'qc_temporal'], output_names=['qc'],
        function=_merge_dicts), name='merge_qc_measures')

    # Plots
    plot_mean = pe.Node(PlotMosaic(title='Mean fMRI'), name='plot_mean')
    plot_tsnr = pe.Node(PlotMosaic(title='tSNR volume'), name='plot_tSNR')
    plot_fd = pe.Node(PlotFD(), name='plot_fd')
    merg = pe.Node(niu.Merge(3), name='plot_metadata')

    workflow.connect([
        (dsource, get_idx,   [('functional_scan', 'in_file'),
                              ('start_idx', 'start_idx'),
                              ('stop_idx', 'stop_idx')]),
        (dsource, merg,      [('session_id', 'in1'),
                              ('scan_id', 'in2'),
                              ('site_name', 'in3')]),
        (dsource, hmcwf,     [('functional_scan', 'inputnode.in_file')]),
        (get_idx, hmcwf,     [('start_idx', 'inputnode.start_idx'),
                              ('stop_idx', 'inputnode.stop_idx')]),
        (hmcwf, bmw,         [('outputnode.out_file', 'inputnode.in_file')]),
        (hmcwf, mean,        [('outputnode.out_file', 'in_file')]),
        (hmcwf, tsnr,        [('outputnode.out_file', 'in_file')]),
        (hmcwf, fd,          [('outputnode.out_xfms', 'in_file')]),
        (mean, plot_mean,    [('out_file', 'in_file')]),
        (tsnr, plot_tsnr,    [('tsnr_file', 'in_file')]),
        (fd, plot_fd,        [('out_file', 'in_file')]),
        (dsource, plot_mean, [('subject_id', 'subject')]),
        (dsource, plot_tsnr, [('subject_id', 'subject')]),
        (dsource, plot_fd,   [('subject_id', 'subject')]),
        (merg, plot_mean,    [('out', 'metadata')]),
        (merg, plot_tsnr,    [('out', 'metadata')]),
        (merg, plot_fd,      [('out', 'metadata')]),
        (bmw, m_spatial,     [('outputnode.out_file', 'func_brain_mask')]),
        (mean, m_spatial,    [('out_file', 'mean_epi')]),
        (dsource, m_spatial, [('subject_id', 'subject_id'),
                              ('session_id', 'session_id'),
                              ('scan_id', 'scan_id'),
                              (('site_name', _empty), 'site_name')]),
        (hmcwf, m_temp,      [('outputnode.out_file', 'func_motion_correct')]),
        (bmw, m_temp,        [('outputnode.out_file', 'func_brain_mask')]),
        (dsource, m_temp,    [('subject_id', 'subject_id'),
                              ('session_id', 'session_id'),
                              ('scan_id', 'scan_id'),
                              (('site_name', _empty), 'site_name')]),
        (fd, m_temp,         [('out_file', 'fd_file')]),
        (tsnr, m_temp,       [('tsnr_file', 'tsnr_volume')]),
        (m_spatial, mqc,     [('qc', 'qc_spatial')]),
        (m_temp, mqc,        [('qc', 'qc_temporal')])
    ])

    if settings.get('mosaic_mask', False):
        workflow.connect(bmw, 'outputnode.out_file', plot_mean, 'in_mask')
        workflow.connect(bmw, 'outputnode.out_file', plot_tsnr, 'in_mask')

    # Export to CSV
    out_csv = op.join(settings['output_dir'], 'fMRIQC.csv')
    to_csv = pe.Node(nam.AddCSVRow(in_file=out_csv), name='write_csv')
    workflow.connect([
        (mqc, to_csv,        [('qc', '_outputs')]),
        (to_csv, outputnode, [('csv_file', 'out_csv')])
    ])

    # Save mean mosaic to well-formed path
    mvmean = pe.Node(niu.Rename(
        format_string='meanepi_%(subject_id)s_%(session_id)s_%(scan_id)s',
        keep_ext=True), name='rename_mean_mosaic')
    dsmean = pe.Node(nio.DataSink(
        base_directory=settings['work_dir'], parameterization=False),
        name='ds_mean')
    workflow.connect([
        (dsource, mvmean,   [('subject_id', 'subject_id'),
                             ('session_id', 'session_id'),
                             ('scan_id', 'scan_id')]),
        (plot_mean, mvmean, [('out_file', 'in_file')]),
        (mvmean, dsmean,    [('out_file', '@mosaic')]),
        (dsource, dsmean,   [('subject_id', 'container')])
    ])

    # Save tSNR mosaic to well-formed path
    mvtsnr = pe.Node(niu.Rename(
        format_string='tsnr_%(subject_id)s_%(session_id)s_%(scan_id)s',
        keep_ext=True), name='rename_tsnr_mosaic')
    dstsnr = pe.Node(nio.DataSink(
        base_directory=settings['work_dir'], parameterization=False),
        name='ds_tsnr')
    workflow.connect([
        (dsource, mvtsnr,   [('subject_id', 'subject_id'),
                             ('session_id', 'session_id'),
                             ('scan_id', 'scan_id')]),
        (plot_tsnr, mvtsnr, [('out_file', 'in_file')]),
        (mvtsnr, dstsnr,    [('out_file', '@mosaic')]),
        (dsource, dstsnr,   [('subject_id', 'container')])
    ])

    # Save FD plot to well-formed path
    mvfd = pe.Node(niu.Rename(
        format_string='fd_%(subject_id)s_%(session_id)s_%(scan_id)s',
        keep_ext=True), name='rename_fd_mosaic')
    dsfd = pe.Node(nio.DataSink(
        base_directory=settings['work_dir'], parameterization=False),
        name='ds_fd')
    workflow.connect([
        (dsource, mvfd, [('subject_id', 'subject_id'),
                         ('session_id', 'session_id'),
                         ('scan_id', 'scan_id')]),
        (plot_fd, mvfd, [('out_file', 'in_file')]),
        (mvfd, dsfd,    [('out_file', '@mosaic')]),
        (dsource, dsfd, [('subject_id', 'container')])
    ])

    return workflow, out_csv


def fmri_bmsk_workflow(name='fMRIBrainMask', use_bet=False):
    """Comute brain mask of an fmri dataset"""

    wf = pe.Workflow(name=name)
    inputnode = pe.Node(niu.IdentityInterface(fields=['in_file']),
                        name='inputnode')
    outputnode = pe.Node(niu.IdentityInterface(fields=['out_file']),
                         name='outputnode')

    if not use_bet:
        afni_msk = pe.Node(afp.Automask(
            outputtype='NIFTI_GZ'), name='afni_msk')

        # Connect brain mask extraction
        wf.connect([
            (inputnode, afni_msk, [('in_file', 'in_file')]),
            (afni_msk, outputnode, [('out_file', 'out_file')])
        ])

    else:
        from nipype.interfaces.fsl import BET, ErodeImage
        bet_msk = pe.Node(BET(mask=True, functional=True), name='bet_msk')
        erode = pe.Node(ErodeImage(kernel_shape='box', kernel_size=1.0),
                        name='erode')

        # Connect brain mask extraction
        wf.connect([
            (inputnode, bet_msk, [('in_file', 'in_file')]),
            (bet_msk, erode,     [('mask_file', 'in_file')]),
            (erode, outputnode,  [('out_file', 'out_file')])
        ])

    return wf


def fmri_hmc_workflow(name='fMRI_HMC', st_correct=False):
    """A head motion correction (HMC) workflow for functional scans"""

    wf = pe.Workflow(name=name)
    inputnode = pe.Node(niu.IdentityInterface(
        fields=['in_file', 'start_idx', 'stop_idx']), name='inputnode')
    outputnode = pe.Node(niu.IdentityInterface(
        fields=['out_file', 'out_xfms']), name='outputnode')

    drop_trs = pe.Node(afp.Calc(expr='a', outputtype='NIFTI_GZ'),
                       name='drop_trs')
    deoblique = pe.Node(afp.Refit(deoblique=True), name='deoblique')
    reorient = pe.Node(afp.Resample(
        orientation='RPI', outputtype='NIFTI_GZ'), name='reorient')
    get_mean_RPI = pe.Node(afp.TStat(
        options='-mean', outputtype='NIFTI_GZ'), name='get_mean_RPI')

    # calculate hmc parameters
    hmc = pe.Node(
        afp.Volreg(args='-Fourier -twopass', zpad=4, outputtype='NIFTI_GZ'),
        name='motion_correct')

    get_mean_motion = get_mean_RPI.clone('get_mean_motion')
    hmc_A = hmc.clone('motion_correct_A')
    hmc_A.inputs.md1d_file = 'max_displacement.1D'

    wf.connect([
        (inputnode, drop_trs,    [('in_file', 'in_file_a'),
                                  ('start_idx', 'start_idx'),
                                  ('stop_idx', 'stop_idx')]),
        (deoblique, reorient,    [('out_file', 'in_file')]),
        (reorient, get_mean_RPI, [('out_file', 'in_file')]),
        (reorient, hmc,          [('out_file', 'in_file')]),
        (get_mean_RPI, hmc,      [('out_file', 'basefile')]),
        (hmc, get_mean_motion,   [('out_file', 'in_file')]),
        (reorient, hmc_A,        [('out_file', 'in_file')]),
        (get_mean_motion, hmc_A, [('out_file', 'basefile')]),
        (hmc_A, outputnode,      [('out_file', 'out_file'),
                                  ('oned_matrix_save', 'out_xfms')])
    ])

    if st_correct:
        st_corr = pe.Node(afp.TShift(outputtype='NIFTI_GZ'), name='TimeShifts')
        wf.connect([
            (drop_trs, st_corr, [('out_file', 'in_file')]),
            (st_corr, deoblique, [('out_file', 'in_file')])
        ])
    else:
        wf.connect([
            (drop_trs, deoblique, [('out_file', 'in_file')])
        ])

    return wf
