#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# @Author: oesteban
# @Date:   2016-01-05 16:15:08
# @Email:  code@oscaresteban.es
# @Last modified by:   oesteban
# @Last Modified time: 2016-09-16 16:28:38
""" A QC workflow for fMRI data """
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from __future__ import unicode_literals
import os
import os.path as op

from nipype.pipeline import engine as pe
from nipype.algorithms import confounds as nac
from nipype.interfaces import io as nio
from nipype.interfaces import utility as niu
from nipype.interfaces import fsl
from nipype.interfaces.afni import preprocess as afp

from mriqc.workflows.utils import fmri_getidx, fwhm_dict
from mriqc.interfaces.qc import FunctionalQC
from mriqc.interfaces.viz import PlotMosaic, PlotFD
from mriqc.utils.misc import bids_getfile, bids_path


def fmri_qc_workflow(name='fMRIQC', settings=None):
    """ The fMRI qc workflow """

    if settings is None:
        settings = {}

    workflow = pe.Workflow(name=name)
    deriv_dir = op.abspath(op.join(settings['output_dir'], 'derivatives'))

    if not op.exists(deriv_dir):
        os.makedirs(deriv_dir)

    # Read FD radius, or default it
    fd_radius = settings.get('fd_radius', 50.)

    # Define workflow, inputs and outputs
    inputnode = pe.Node(niu.IdentityInterface(
        fields=['bids_dir', 'subject_id', 'session_id', 'run_id',
                'site_name', 'start_idx', 'stop_idx']), name='inputnode')
    get_idx = pe.Node(niu.Function(
        input_names=['in_file', 'start_idx', 'stop_idx'], function=fmri_getidx,
        output_names=['start_idx', 'stop_idx']), name='get_idx')

    outputnode = pe.Node(niu.IdentityInterface(
        fields=['qc', 'mosaic', 'out_group', 'out_movpar', 'out_dvars',
                'out_fd']), name='outputnode')


    # 0. Get data
    datasource = pe.Node(niu.Function(
        input_names=['bids_dir', 'data_type', 'subject_id', 'session_id', 'run_id'],
        output_names=['out_file'], function=bids_getfile), name='datasource')
    datasource.inputs.data_type = 'func'

    # Workflow --------------------------------------------------------
    # 1. HMC: head motion correct
    hmcwf = hmc_mcflirt()
    if settings.get('hmc_afni', False):
        hmcwf = hmc_afni(st_correct=settings.get('correct_slice_timing', False))
    hmcwf.inputs.inputnode.fd_radius = fd_radius

    mean = pe.Node(afp.TStat(                   # 2. Compute mean fmri
        options='-mean', outputtype='NIFTI_GZ'), name='mean')
    bmw = fmri_bmsk_workflow(                   # 3. Compute brain mask
        use_bet=settings.get('use_bet', False))

    # Compute TSNR using nipype implementation
    tsnr = pe.Node(nac.TSNR(), name='compute_tsnr')

    # Compute DVARS
    dvnode = pe.Node(nac.ComputeDVARS(remove_zerovariance=True, save_plot=True,
                     save_all=True, figdpi=200, figformat='pdf'), name='ComputeDVARS')
    fdnode = pe.Node(nac.FramewiseDisplacement(
        normalize=True, save_plot=True, radius=fd_radius,
        figdpi=200), name='ComputeFD')

    # AFNI quality measures
    fwhm = pe.Node(afp.FWHMx(combine=True, detrend=True), name='smoothness')
    # fwhm.inputs.acf = True  # add when AFNI >= 16
    outliers = pe.Node(afp.OutlierCount(fraction=True, out_file='ouliers.out'),
                       name='outliers')
    quality = pe.Node(afp.QualityIndex(automask=True), out_file='quality.out',
                      name='quality')

    measures = pe.Node(FunctionalQC(), name='measures')

    # Link images that should be reported
    dsreport = pe.Node(nio.DataSink(
        base_directory=settings['report_dir'], parameterization=True), name='dsreport')
    dsreport.inputs.container = 'func'
    dsreport.inputs.substitutions = [
        ('_data', ''),
        ('fd_power_2012', 'plot_fd'),
        ('tsnr.nii.gz', 'mosaic_TSNR.nii.gz'),
        ('mean.nii.gz', 'mosaic_TSNR_mean.nii.gz'),
        ('stdev.nii.gz', 'mosaic_TSNR_stdev.nii.gz')
    ]
    dsreport.inputs.regexp_substitutions = [
        ('_u?(sub-[\\w\\d]*)\\.([\\w\\d_]*)(?:\\.([\\w\\d_-]*))+', '\\1_ses-\\2_\\3'),
        ('sub-[^/.]*_dvars_std', 'plot_dvars'),
        ('sub-[^/.]*_mask', 'mask'),
        ('sub-[^/.]*_mcf_tstat', 'mosaic_epi_mean')
    ]

    workflow.connect([
        (inputnode, datasource, [('bids_dir', 'bids_dir'),
                                 ('subject_id', 'subject_id'),
                                 ('session_id', 'session_id'),
                                 ('run_id', 'run_id')]),
        (inputnode, get_idx, [('start_idx', 'start_idx'),
                              ('stop_idx', 'stop_idx')]),
        (datasource, get_idx, [('out_file', 'in_file')]),
        (datasource, hmcwf, [('out_file', 'inputnode.in_file')]),
        (get_idx, hmcwf, [('start_idx', 'inputnode.start_idx'),
                          ('stop_idx', 'inputnode.stop_idx')]),
        (hmcwf, bmw, [('outputnode.out_file', 'inputnode.in_file')]),
        (hmcwf, mean, [('outputnode.out_file', 'in_file')]),
        (hmcwf, tsnr, [('outputnode.out_file', 'in_file')]),
        (hmcwf, fdnode, [('outputnode.out_movpar', 'in_plots')]),
        (mean, fwhm, [('out_file', 'in_file')]),
        (bmw, fwhm, [('outputnode.out_file', 'mask')]),
        (hmcwf, outliers, [('outputnode.out_file', 'in_file')]),
        (bmw, outliers, [('outputnode.out_file', 'mask')]),
        (hmcwf, quality, [('outputnode.out_file', 'in_file')]),
        (hmcwf, dvnode, [('outputnode.out_file', 'in_file')]),
        (bmw, dvnode, [('outputnode.out_file', 'in_mask')]),
        (mean, measures, [('out_file', 'in_epi')]),
        (hmcwf, measures, [('outputnode.out_file', 'in_hmc')]),
        (bmw, measures, [('outputnode.out_file', 'in_mask')]),
        (tsnr, measures, [('tsnr_file', 'in_tsnr')]),
        (dvnode, measures, [('out_all', 'in_dvars')]),
        (fdnode, measures, [('out_file', 'in_fd')]),
        (fdnode, outputnode, [('out_file', 'out_fd')]),
        (dvnode, outputnode, [('out_all', 'out_dvars')]),
        (hmcwf, outputnode, [('outputnode.out_movpar', 'out_movpar')]),
        (mean, dsreport, [('out_file', '@meanepi')]),
        (tsnr, dsreport, [('tsnr_file', '@tsnr'),
                          ('stddev_file', '@tsnr_std'),
                          ('mean_file', '@tsnr_mean')]),
        (bmw, dsreport, [('outputnode.out_file', '@mask')]),
        (fdnode, dsreport, [('out_figure', '@fdplot')]),
        (dvnode, dsreport, [('fig_std', '@dvars')]),
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
        (fwhm, datasink, [(('fwhm', fwhm_dict), 'fwhm')]),
        (outliers, datasink, [(('out_file', _parse_tout), 'outlier')]),
        (quality, datasink, [(('out_file', _parse_tqual), 'quality')]),
        (measures, datasink, [('summary', 'summary'),
                              ('spacing', 'spacing'),
                              ('size', 'size'),
                              ('fber', 'fber'),
                              ('efc', 'efc'),
                              ('snr', 'snr'),
                              ('gsr', 'gsr'),
                              ('m_tsnr', 'm_tsnr'),
                              ('fd', 'fd'),
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

def hmc_mcflirt(name='fMRI_HMC_mcflirt'):
    """
    An :abbr:`HMC (head motion correction)` for functional scans
    using FSL MCFLIRT
    """

    workflow = pe.Workflow(name=name)
    inputnode = pe.Node(niu.IdentityInterface(
        fields=['in_file', 'fd_radius', 'start_idx', 'stop_idx']), name='inputnode')
    outputnode = pe.Node(niu.IdentityInterface(
        fields=['out_file', 'out_movpar']), name='outputnode')

    mcflirt = pe.Node(fsl.MCFLIRT(mean_vol=True, save_plots=True,
                                  save_rms=True, save_mats=True), name="MCFLIRT")

    workflow.connect([
        (inputnode, mcflirt, [('in_file', 'in_file')]),
        (mcflirt, outputnode, [('out_file', 'out_file'),
                               ('par_file', 'out_movpar')])

    ])
    return workflow



def hmc_afni(name='fMRI_HMC_afni', st_correct=False):
    """A head motion correction (HMC) workflow for functional scans"""

    workflow = pe.Workflow(name=name)
    inputnode = pe.Node(niu.IdentityInterface(
        fields=['in_file', 'fd_radius', 'start_idx', 'stop_idx']), name='inputnode')
    outputnode = pe.Node(niu.IdentityInterface(
        fields=['out_file', 'out_movpar']), name='outputnode')

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
                             ('oned_file', 'out_movpar')])
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
