#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# @Author: oesteban
# @Date:   2016-01-05 16:15:08
# @Email:  code@oscaresteban.es
# @Last modified by:   oesteban
# @Last Modified time: 2016-05-05 15:09:56
""" A QC workflow for fMRI data """
import os
import os.path as op

from nipype.pipeline import engine as pe
from nipype.algorithms import misc as nam
from nipype.interfaces import io as nio
from nipype.interfaces import utility as niu
from nipype.interfaces.afni import preprocess as afp
from .utils import fmri_getidx, fwhm_dict
from ..interfaces.qc import FunctionalQC, FramewiseDisplacement
from ..interfaces.viz import PlotMosaic, PlotFD
from ..utils.misc import bids_getfile, bids_path


def fmri_qc_workflow(name='fMRIQC', settings=None):
    """ The fMRI qc workflow """

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
        fields=['bids_root', 'subject_id', 'session_id', 'run_id',
                'site_name', 'start_idx', 'stop_idx']), name='inputnode')
    get_idx = pe.Node(niu.Function(
        input_names=['in_file', 'start_idx', 'stop_idx'], function=fmri_getidx,
        output_names=['start_idx', 'stop_idx']), name='get_idx')

    outputnode = pe.Node(niu.IdentityInterface(
        fields=['qc', 'mosaic', 'out_group']), name='outputnode')


    # 0. Get data
    datasource = pe.Node(niu.Function(
        input_names=['bids_root', 'data_type', 'subject_id', 'session_id', 'run_id'],
        output_names=['out_file'], function=bids_getfile), name='datasource')
    datasource.inputs.data_type = 'func'

    # Workflow --------------------------------------------------------
    hmcwf = fmri_hmc_workflow(                  # 1. HMC: head motion correct
        st_correct=settings.get('correct_slice_timing', False))
    mean = pe.Node(afp.TStat(                   # 2. Compute mean fmri
        options='-mean', outputtype='NIFTI_GZ'), name='mean')
    bmw = fmri_bmsk_workflow(                   # 3. Compute brain mask
        use_bet=settings.get('use_bet', False))

    fdisp = pe.Node(FramewiseDisplacement(), name='generate_FD_file')
    tsnr = pe.Node(nam.TSNR(), name='compute_tsnr')

    # AFNI quality measures
    fwhm = pe.Node(afp.FWHMx(combine=True, detrend=True), name='smoothness')
    # fwhm.inputs.acf = True  # add when AFNI >= 16
    outliers = pe.Node(afp.OutlierCount(fraction=True, out_file='ouliers.out'),
                       name='outliers')
    quality = pe.Node(afp.QualityIndex(automask=True), out_file='quality.out',
                      name='quality')

    measures = pe.Node(FunctionalQC(), name='measures')

    # Plots
    plot_mean = pe.Node(PlotMosaic(title='Mean fMRI'), name='plot_mean')
    plot_tsnr = pe.Node(PlotMosaic(title='tSNR volume'), name='plot_tSNR')
    plot_fd = pe.Node(PlotFD(), name='plot_fd')
    merg = pe.Node(niu.Merge(3), name='plot_metadata')

    workflow.connect([
        (inputnode, datasource, [('bids_root', 'bids_root'),
                                 ('subject_id', 'subject_id'),
                                 ('session_id', 'session_id'),
                                 ('run_id', 'run_id')]),
        (inputnode, get_idx, [('start_idx', 'start_idx'),
                              ('stop_idx', 'stop_idx')]),
        (datasource, get_idx, [('out_file', 'in_file')]),
        (inputnode, merg, [('session_id', 'in1'),
                           ('run_id', 'in2'),
                           ('site_name', 'in3')]),
        (datasource, hmcwf, [('out_file', 'inputnode.in_file')]),
        (get_idx, hmcwf, [('start_idx', 'inputnode.start_idx'),
                          ('stop_idx', 'inputnode.stop_idx')]),
        (hmcwf, bmw, [('outputnode.out_file', 'inputnode.in_file')]),
        (hmcwf, mean, [('outputnode.out_file', 'in_file')]),
        (hmcwf, tsnr, [('outputnode.out_file', 'in_file')]),
        (hmcwf, fdisp, [('outputnode.out_xfms', 'in_file')]),
        (mean, plot_mean, [('out_file', 'in_file')]),
        (tsnr, plot_tsnr, [('tsnr_file', 'in_file')]),
        (fdisp, plot_fd, [('out_file', 'in_file')]),
        (inputnode, plot_mean, [('subject_id', 'subject')]),
        (inputnode, plot_tsnr, [('subject_id', 'subject')]),
        (inputnode, plot_fd, [('subject_id', 'subject')]),
        (merg, plot_mean, [('out', 'metadata')]),
        (merg, plot_tsnr, [('out', 'metadata')]),
        (merg, plot_fd, [('out', 'metadata')]),
        (mean, fwhm, [('out_file', 'in_file')]),
        (bmw, fwhm, [('outputnode.out_file', 'mask')]),
        (hmcwf, outliers, [('outputnode.out_file', 'in_file')]),
        (bmw, outliers, [('outputnode.out_file', 'mask')]),
        (hmcwf, quality, [('outputnode.out_file', 'in_file')]),
        (mean, measures, [('out_file', 'in_epi')]),
        (hmcwf, measures, [('outputnode.out_file', 'in_hmc')]),
        (bmw, measures, [('outputnode.out_file', 'in_mask')]),
        (tsnr, measures, [('tsnr_file', 'in_tsnr')])
    ])

    if settings.get('mosaic_mask', False):
        workflow.connect(bmw, 'outputnode.out_file', plot_mean, 'in_mask')
        workflow.connect(bmw, 'outputnode.out_file', plot_tsnr, 'in_mask')

    # Save mean mosaic to well-formed path
    mvmean = pe.Node(niu.Rename(
        format_string='meanepi_%(subject_id)s_%(session_id)s_%(run_id)s',
        keep_ext=True), name='rename_mean_mosaic')
    dsmean = pe.Node(nio.DataSink(base_directory=settings['work_dir'], parameterization=False),
                     name='ds_mean')
    workflow.connect([
        (inputnode, mvmean, [('subject_id', 'subject_id'),
                           ('session_id', 'session_id'),
                           ('run_id', 'run_id')]),
        (plot_mean, mvmean, [('out_file', 'in_file')]),
        (mvmean, dsmean, [('out_file', '@mosaic')])
    ])
    # Save tSNR mosaic to well-formed path
    mvtsnr = pe.Node(niu.Rename(
        format_string='tsnr_%(subject_id)s_%(session_id)s_%(run_id)s',
        keep_ext=True), name='rename_tsnr_mosaic')
    dstsnr = pe.Node(nio.DataSink(base_directory=settings['work_dir'], parameterization=False),
                     name='ds_tsnr')
    workflow.connect([
        (inputnode, mvtsnr, [('subject_id', 'subject_id'),
                           ('session_id', 'session_id'),
                           ('run_id', 'run_id')]),
        (plot_tsnr, mvtsnr, [('out_file', 'in_file')]),
        (mvtsnr, dstsnr, [('out_file', '@mosaic')])
    ])
    # Save FD plot to well-formed path
    mvfd = pe.Node(niu.Rename(
        format_string='fd_%(subject_id)s_%(session_id)s_%(run_id)s',
        keep_ext=True), name='rename_fd_mosaic')
    dsfd = pe.Node(nio.DataSink(base_directory=settings['work_dir'], parameterization=False),
                   name='ds_fd')
    workflow.connect([
        (inputnode, mvfd, [('subject_id', 'subject_id'),
                         ('session_id', 'session_id'),
                         ('run_id', 'run_id')]),
        (plot_fd, mvfd, [('out_file', 'in_file')]),
        (mvfd, dsfd, [('out_file', '@mosaic')])
    ])

    # Format name
    out_name = pe.Node(niu.Function(
        input_names=['subid', 'sesid', 'runid', 'prefix', 'out_path'], output_names=['out_file'],
        function=bids_path), name='FormatName')
    out_name.inputs.out_path = deriv_dir
    out_name.inputs.prefix = 'func'

    # Save to JSON file
    datasink = pe.Node(nio.JSONFileSink(), name='datasink')
    datasink.inputs.qc_type = 'func'

    workflow.connect([
        (inputnode, out_name, [('subject_id', 'subid'),
                               ('session_id', 'sesid'),
                               ('run_id', 'runid')]),
        (inputnode, datasink, [('subject_id', 'subject_id'),
                               ('session_id', 'session_id'),
                               ('run_id', 'run_id')]),
        (plot_mean, datasink, [('out_file', 'mean_plot')]),
        (plot_tsnr, datasink, [('out_file', 'tsnr_plot')]),
        (plot_fd, datasink, [('out_file', 'fd_plot')]),
        (fwhm, datasink, [(('fwhm', fwhm_dict), 'fwhm')]),
        (outliers, datasink, [(('out_file', _parse_tout), 'outlier')]),
        (quality, datasink, [(('out_file', _parse_tqual), 'quality')]),
        (fdisp, datasink, [('fd_stats', 'fd_stats')]),
        (measures, datasink, [('summary', 'summary'),
                              ('spacing', 'spacing'),
                              ('size', 'size'),
                              ('fber', 'fber'),
                              ('efc', 'efc'),
                              ('snr', 'snr'),
                              ('gsr', 'gsr'),
                              ('m_tsnr', 'm_tsnr'),
                              ('dvars', 'dvars'),
                              ('gcor', 'gcor')]),
        (out_name, datasink, [('out_file', 'out_file')]),
        (datasink, outputnode, [('out_file', 'out_file')])
    ])

    return workflow


def fmri_bmsk_workflow(name='fMRIBrainMask', use_bet=False):
    """Comute brain mask of an fmri dataset"""

    workflow = pe.Workflow(name=name)
    inputnode = pe.Node(niu.IdentityInterface(fields=['in_file']),
                        name='inputnode')
    outputnode = pe.Node(niu.IdentityInterface(fields=['out_file']),
                         name='outputnode')

    if not use_bet:
        afni_msk = pe.Node(afp.Automask(
            outputtype='NIFTI_GZ'), name='afni_msk')

        # Connect brain mask extraction
        workflow.connect([
            (inputnode, afni_msk, [('in_file', 'in_file')]),
            (afni_msk, outputnode, [('out_file', 'out_file')])
        ])

    else:
        from nipype.interfaces.fsl import BET, ErodeImage
        bet_msk = pe.Node(BET(mask=True, functional=True), name='bet_msk')
        erode = pe.Node(ErodeImage(kernel_shape='box', kernel_size=1.0),
                        name='erode')

        # Connect brain mask extraction
        workflow.connect([
            (inputnode, bet_msk, [('in_file', 'in_file')]),
            (bet_msk, erode, [('mask_file', 'in_file')]),
            (erode, outputnode, [('out_file', 'out_file')])
        ])

    return workflow


def fmri_hmc_workflow(name='fMRI_HMC', st_correct=False):
    """A head motion correction (HMC) workflow for functional scans"""

    workflow = pe.Workflow(name=name)
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

    workflow.connect([
        (inputnode, drop_trs, [('in_file', 'in_file_a'),
                               ('start_idx', 'start_idx'),
                               ('stop_idx', 'stop_idx')]),
        (deoblique, reorient, [('out_file', 'in_file')]),
        (reorient, get_mean_RPI, [('out_file', 'in_file')]),
        (reorient, hmc, [('out_file', 'in_file')]),
        (get_mean_RPI, hmc, [('out_file', 'basefile')]),
        (hmc, get_mean_motion, [('out_file', 'in_file')]),
        (reorient, hmc_A, [('out_file', 'in_file')]),
        (get_mean_motion, hmc_A, [('out_file', 'basefile')]),
        (hmc_A, outputnode, [('out_file', 'out_file'),
                             ('oned_matrix_save', 'out_xfms')])
    ])

    if st_correct:
        st_corr = pe.Node(afp.TShift(outputtype='NIFTI_GZ'), name='TimeShifts')
        workflow.connect([
            (drop_trs, st_corr, [('out_file', 'in_file')]),
            (st_corr, deoblique, [('out_file', 'in_file')])
        ])
    else:
        workflow.connect([
            (drop_trs, deoblique, [('out_file', 'in_file')])
        ])

    return workflow

def _merge_dicts(in_qc, subject_id, metadata, fwhm, fd_stats, outlier, quality):
    in_qc['subject'] = subject_id
    in_qc['session'] = metadata[0]
    in_qc['scan'] = metadata[1]

    try:
        in_qc['site_name'] = metadata[2]
    except IndexError:
        pass  # No site_name defined

    in_qc.update({'fwhm_x': fwhm[0], 'fwhm_y': fwhm[1], 'fwhm_z': fwhm[2],
                  'fwhm': fwhm[3]})
    in_qc.update(fd_stats)
    in_qc['outlier'] = outlier
    in_qc['quality'] = quality

    try:
        in_qc['tr'] = in_qc['spacing_tr']
    except KeyError:
        pass  # TR is not defined

    return in_qc

def _mean(inlist):
    import numpy as np
    return np.mean(inlist)

def _parse_tqual(in_file):
    import numpy as np
    with open(in_file, 'r') as fin:
        lines = fin.readlines()
        # remove general information
        lines = [l for l in lines if l[:2] != "++"]
        # remove general information and warnings
        return np.mean([float(l.strip()) for l in lines])
    raise RuntimeError('AFNI 3dTqual was not parsed correctly')

def _parse_tout(in_file):
    import numpy as np
    data = np.loadtxt(in_file)  # pylint: disable=no-member
    return data.mean()
