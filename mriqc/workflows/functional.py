#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# @Author: oesteban
# @Date:   2016-01-05 16:15:08
# @Email:  code@oscaresteban.es
# @Last modified by:   oesteban
# @Last Modified time: 2016-01-05 18:06:34
import os
import os.path as op
import sys

import nipype.interfaces.io as nio
import nipype.pipeline.engine as pe
import nipype.interfaces.utility as niu
from nipype.interfaces import fsl
from nipype.interfaces.afni import preprocess as afp

from nipype import logging
logger = logging.getLogger('workflow')


def fmri_qc_workflow(name='fMRIQC', settings={}):
    from qap.workflows.utils import qap_functional_spatial as qc_fmri_spat
    from qap.workflows.utils import qap_functional_temporal as qc_fmri_temp
    from qap.temporal_qc import fd_jenkinson
    from ..interfaces.viz import PlotMosaic, PlotFD

    # Define workflow, inputs and outputs
    workflow = pe.Workflow(name=name)
    inputnode = pe.Node(niu.IdentityInterface(
        fields=['in_file', 'subject_id', 'session_id', 'scan_id',
                'site_name']), name='inputnode')
    outputnode = pe.Node(niu.IdentityInterface(
        fields=['qc', 'mosaic']), name='outputnode')

    # Measures
    m_spatial = pe.Node(niu.Function(
        input_names=['mean_epi', 'func_brain_mask', 'subject_id', 'session_id',
                     'scan_id', 'site_name'], output_names=['qc'],
        function=qc_fmri_spat), name='m_spatial')
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

    # Plots
    plot_mean = pe.Node(PlotMosaic(title='Mean fMRI'), name='plot_mean')
    plot_tsnr = pe.Node(PlotMosaic(title='tSNR volume'), name='plot_tSNR')
    fdplot = pe.Node(PlotFD(), name='plot_fd')
    merg = pe.Node(niu.Merge(3), name='plot_metadata')

    workflow.connect([
        (inputnode, merg,  [('session_id', 'in1'),
                            ('scan_id', 'in2'),
                            ('site_name', 'in3')]),
        (inputnode, hmcwf, [('in_file', 'inputnode.in_file'),
                            ('start_idx', 'inputnode.start_idx'),
                            ('stop_idx', 'inputnode.stop_idx')]),
        (hmcwf, bmw,       [('outputnode.out_file', 'inputnode.in_file')]),
        (hmcwf, mean,      [('outputnode.out_file', 'in_file')]),
        (hmcwf, tsnr,      [('outputnode.out_file', 'in_file')]),
        (hmcwf, fd,        [('outputnode.out_xfms', 'in_file')]),
        (hmcwf, plot_mean, [('outputnode.out_file', 'in_file')]),
        (tsnr, plot_tsnr,  [('tsnr_file', 'tsnr_volume')]),
        (fd, fdplot,       [('out_file', 'in_file')]),
        (inputnode, plot_mean, [('subject_id', 'subject')]),
        (inputnode, plot_tsnr, [('subject_id', 'subject')]),
        (inputnode, fdplot, [('subject_id', 'subject')]),
        (merg, plot_mean,  [('out', 'metadata')]),
        (merg, plot_tsnr,  [('out', 'metadata')]),
        (merg, fdplot,     [('out', 'metadata')]),
        (bmw, m_spatial,   [('outputnode.out_file', 'func_brain_mask')]),
        (mean, m_spatial,  [('out_file', 'mean_epi')]),
        (fd, m_temp,       [('out_file', 'fd_file')]),
        (tsnr, m_temp,     [('tsnr_file', 'tsnr_volume')]),
    ])

    if settings.get('mosaic_mask', False):
        workflow.connect(bmw, 'outputnode.out_file', plot_mean, 'in_mask')
        workflow.connect(bmw, 'outputnode.out_file', plot_tsnr, 'in_mask')

    return workflow


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

    from .utils import fmri_getidx

    wf = pe.Workflow(name=name)
    inputnode = pe.Node(niu.IdentityInterface(
        fields=['in_file', 'start_idx', 'stop_idx']), name='inputnode')
    outputnode = pe.Node(niu.IdentityInterface(
        fields=['out_file', 'out_xfms']), name='outputnode')

    get_idx = pe.Node(niu.Function(
        input_names=['in_file', 'start_idx', 'stop_idx'], function=fmri_getidx,
        output_names=['start_idx', 'stop_idx']), name='get_idx')
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
        (inputnode, get_idx,     [('in_file', 'in_file'),
                                  ('start_idx', 'start_idx'),
                                  ('stop_idx', 'stop_idx')]),
        (inputnode, drop_trs,    [('in_file', 'in_file_a')]),
        (get_idx, drop_trs,      [('start_idx', 'start_idx'),
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
