#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# @Author: oesteban
# @Date:   2016-01-05 16:15:08
# @Email:  code@oscaresteban.es
# @Last modified by:   oesteban
# @Last Modified time: 2016-10-10 16:39:57
""" A QC workflow for fMRI data """
from __future__ import print_function, division, absolute_import, unicode_literals
import os
import os.path as op

from nipype.pipeline import engine as pe
from nipype.algorithms import confounds as nac
from nipype.interfaces import io as nio
from nipype.interfaces import utility as niu
from nipype.interfaces import fsl
from nipype.interfaces import freesurfer as fs
from nipype.interfaces.afni import preprocess as afp

from mriqc.workflows.utils import fmri_getidx, fwhm_dict, fd_jenkinson, thresh_image, slice_wise_fft
from mriqc.interfaces.qc import FunctionalQC
from mriqc.interfaces.functional import Spikes
from mriqc.interfaces.viz import PlotMosaic, PlotFD
from mriqc.utils.misc import bids_getfile, bids_path, check_folder

def fmri_qc_workflow(name='fMRIQC', settings=None):
    """ The fMRI qc workflow """

    if settings is None:
        settings = {}

    workflow = pe.Workflow(name=name)
    deriv_dir = check_folder(
        op.abspath(op.join(settings['output_dir'], 'derivatives')))

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

    # 0. Get data, put it in RAS orientation
    datasource = pe.Node(niu.Function(
        input_names=[
            'bids_dir', 'data_type', 'subject_id', 'session_id', 'run_id'],
        output_names=['out_file'], function=bids_getfile), name='datasource')
    datasource.inputs.data_type = 'func'

    reorient = pe.Node(fs.MRIConvert(out_type='niigz', out_orientation='RAS'),
                       name='EPIReorient')

    # Workflow --------------------------------------------------------
    # 1. HMC: head motion correct
    hmcwf = hmc_mcflirt()
    if settings.get('hmc_afni', False):
        hmcwf = hmc_afni(
            st_correct=settings.get('correct_slice_timing', False))
    hmcwf.inputs.inputnode.fd_radius = fd_radius

    mean = pe.Node(afp.TStat(                   # 2. Compute mean fmri
        options='-mean', outputtype='NIFTI_GZ'), name='mean')
    bmw = fmri_bmsk_workflow(                   # 3. Compute brain mask
        use_bet=settings.get('use_bet', False))

    # EPI to MNI registration
    ema = epi_mni_align()

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

    spmask = pe.Node(niu.Function(
        input_names=['in_file', 'in_mask'], output_names=['out_file', 'out_plot'],
        function=spikes_mask), name='SpikesMask')
    spikes = pe.Node(Spikes(), name='SpikesFinder')
    spikes_bg = pe.Node(Spikes(no_zscore=True), name='SpikesFinderBgMask')
    spikes_fft = pe.Node(niu.Function(
        input_names=['in_file'], output_names=['out_fft', 'out_energy', 'out_spikes'],
        function=slice_wise_fft), name='SpikesFinderFFT')

    bigplot = pe.Node(niu.Function(
        input_names=['in_func', 'in_mask', 'in_segm', 'in_spikes',
                     'in_spikes_bg', 'in_spikes_fft', 'fd', 'dvars'],
        output_names=['out_file'], function=_big_plot), name='BigPlot')

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
        ('stdev.nii.gz', 'mosaic_stdev.nii.gz')
    ]
    dsreport.inputs.regexp_substitutions = [
        ('_u?(sub-[\\w\\d]*)\\.([\\w\\d_]*)(?:\\.([\\w\\d_-]*))+',
         '\\1_ses-\\2_\\3'),
        ('sub-[^/.]*_fmriplot', 'plot_fmri'),
        ('sub-[^/.]*_mask', 'mask'),
        ('sub-[^/.]*_mcf_tstat', 'mosaic_epi_mean'),
        ('sub-[^/.]*_spmask', 'plot_spikes_mask'),
    ]

    workflow.connect([
        (inputnode, datasource, [('bids_dir', 'bids_dir'),
                                 ('subject_id', 'subject_id'),
                                 ('session_id', 'session_id'),
                                 ('run_id', 'run_id')]),
        (inputnode, get_idx, [('start_idx', 'start_idx'),
                              ('stop_idx', 'stop_idx')]),
        (datasource, get_idx, [('out_file', 'in_file')]),
        (datasource, reorient, [('out_file', 'in_file')]),
        (reorient, hmcwf, [('out_file', 'inputnode.in_file')]),
        (datasource, spikes, [('out_file', 'in_file')]),
        (datasource, spikes_fft, [('out_file', 'in_file')]),
        (datasource, spikes_bg, [('out_file', 'in_file')]),
        (reorient, bigplot, [('out_file', 'in_func')]),
        (get_idx, hmcwf, [('start_idx', 'inputnode.start_idx'),
                          ('stop_idx', 'inputnode.stop_idx')]),
        (hmcwf, bmw, [('outputnode.out_file', 'inputnode.in_file')]),
        (hmcwf, mean, [('outputnode.out_file', 'in_file')]),
        (hmcwf, tsnr, [('outputnode.out_file', 'in_file')]),
        (hmcwf, fdnode, [('outputnode.out_movpar', 'in_plots')]),
        (hmcwf, spmask, [('outputnode.out_file', 'in_file')]),
        (mean, fwhm, [('out_file', 'in_file')]),
        (bmw, fwhm, [('outputnode.out_file', 'mask')]),
        (bmw, spmask, [('outputnode.out_file', 'in_mask')]),
        (bmw, spikes, [('outputnode.out_file', 'in_mask')]),
        (mean, ema, [('out_file', 'inputnode.epi_mean')]),
        (bmw, ema, [('outputnode.out_file', 'inputnode.epi_mask')]),
        (hmcwf, outliers, [('outputnode.out_file', 'in_file')]),
        (bmw, outliers, [('outputnode.out_file', 'mask')]),
        (hmcwf, quality, [('outputnode.out_file', 'in_file')]),
        (hmcwf, dvnode, [('outputnode.out_file', 'in_file')]),
        (bmw, dvnode, [('outputnode.out_file', 'in_mask')]),
        (bmw, bigplot, [('outputnode.out_file', 'in_mask')]),
        (mean, measures, [('out_file', 'in_epi')]),
        (hmcwf, measures, [('outputnode.out_file', 'in_hmc')]),
        (bmw, measures, [('outputnode.out_file', 'in_mask')]),
        (tsnr, measures, [('tsnr_file', 'in_tsnr')]),
        (dvnode, measures, [('out_all', 'in_dvars')]),
        (fdnode, measures, [('out_file', 'in_fd')]),
        (fdnode, outputnode, [('out_file', 'out_fd')]),
        (fdnode, bigplot, [('out_file', 'fd')]),
        (dvnode, outputnode, [('out_all', 'out_dvars')]),
        (dvnode, bigplot, [('out_std', 'dvars')]),
        (ema, bigplot, [('outputnode.epi_parc', 'in_segm')]),
        (spikes, bigplot, [('out_tsz', 'in_spikes')]),
        (spikes_bg, bigplot, [('out_tsz', 'in_spikes_bg')]),
        (spikes_fft, bigplot, [('out_energy', 'in_spikes_fft')]),
        (hmcwf, outputnode, [('outputnode.out_movpar', 'out_movpar')]),
        (mean, dsreport, [('out_file', '@meanepi')]),
        (spmask, spikes_bg, [('out_file', 'in_mask')]),
        (spmask, dsreport, [('out_plot', '@spmaskplot')]),
        (tsnr, dsreport, [('tsnr_file', '@tsnr'),
                          ('stddev_file', '@tsnr_std'),
                          ('mean_file', '@tsnr_mean')]),
        (bmw, dsreport, [('outputnode.out_file', '@mask')]),
        (bigplot, dsreport, [('out_file', '@fmriplot')]),
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

    mcflirt = pe.Node(fsl.MCFLIRT(save_plots=True, save_rms=True, save_mats=True),
                      name="MCFLIRT")

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

    movpar = pe.Node(niu.Function(
        function=fd_jenkinson, input_names=['in_file', 'rmax'],
        output_names=['out_file']), name='Mat2Movpar')

    workflow.connect([
        (inputnode, drop_trs, [('in_file', 'in_file_a'),
                               ('start_idx', 'start_idx'),
                               ('stop_idx', 'stop_idx')]),
        (inputnode, movpar, [('fd_radius', 'rmax')]),
        (deoblique, reorient, [('out_file', 'in_file')]),
        (reorient, get_mean_RPI, [('out_file', 'in_file')]),
        (reorient, hmc, [('out_file', 'in_file')]),
        (get_mean_RPI, hmc, [('out_file', 'basefile')]),
        (hmc, get_mean_motion, [('out_file', 'in_file')]),
        (reorient, hmc_A, [('out_file', 'in_file')]),
        (get_mean_motion, hmc_A, [('out_file', 'basefile')]),
        (hmc_A, outputnode, [('out_file', 'out_file')]),
        (hmc_A, movpar, [('oned_matrix_save', 'in_file')]),
        (movpar, outputnode, [('out_file', 'out_movpar')])
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


def epi_mni_align(name='SpatialNormalization', settings=None):
    """
    Uses FSL FLIRT with the BBR cost function to find the transform that
    maps the EPI space into the MNI152-nonlinear-symmetric atlas.

    The input epi_mean is the averaged and brain-masked EPI timeseries

    Returns the EPI mean resampled in MNI space (for checking out registration) and
    the associated "lobe" parcellation in EPI space.

    """
    from niworkflows.data import get_mni_icbm152_nlin_sym_09c_las as get_template
    mni_template = get_template()

    workflow = pe.Workflow(name=name)
    inputnode = pe.Node(niu.IdentityInterface(fields=['epi_mean', 'epi_mask']),
                        name='inputnode')
    outputnode = pe.Node(niu.IdentityInterface(fields=['epi_mni', 'epi_parc']),
                         name='outputnode')

    # Mask PD template image
    brainmask = pe.Node(fsl.ApplyMask(
        in_file=op.join(
            mni_template, 'mni_icbm152_pd_tal_nlin_sym_09c.nii.gz'),
        mask_file=op.join(mni_template, 'mni_icbm152_t1_tal_nlin_sym_09c_mask.nii.gz')),
        name='MNIApplyMask')

    epimask = pe.Node(fsl.ApplyMask(), name='EPIApplyMask')

    # Extract wm mask from segmentation
    wm_mask = pe.Node(niu.Function(input_names=['in_file'], output_names=['out_file'],
                                   function=thresh_image), name='WM_mask')
    wm_mask.inputs.in_file = op.join(
        mni_template, 'mni_icbm152_wm_tal_nlin_sym_09c.nii.gz')

    flt_bbr_init = pe.Node(fsl.FLIRT(dof=12, out_matrix_file='init.mat'),
                           name='Flirt_BBR_init')
    flt_bbr = pe.Node(fsl.FLIRT(dof=12, cost_func='bbr'), name='Flirt_BBR')
    flt_bbr.inputs.schedule = op.join(os.getenv('FSLDIR'),
                                      'etc/flirtsch/bbr.sch')

    # make equivalent warp fields
    invt_bbr = pe.Node(fsl.ConvertXFM(invert_xfm=True), name='Flirt_BBR_Inv')
    # Warp segmentation into EPI space
    segm_xfm = pe.Node(fsl.ApplyXfm(
        in_file=op.join(mni_template, 'atlasgrey.nii.gz'),
        interp='nearestneighbour'), name='ResampleSegmentation')

    workflow.connect([
        (inputnode, epimask, [('epi_mean', 'in_file'),
                              ('epi_mask', 'mask_file')]),
        (epimask, flt_bbr_init, [('out_file', 'in_file')]),
        (epimask, flt_bbr, [('out_file', 'in_file')]),
        (brainmask, flt_bbr_init, [('out_file', 'reference')]),
        (brainmask, flt_bbr, [('out_file', 'reference')]),
        (wm_mask, flt_bbr, [('out_file', 'wm_seg')]),
        (flt_bbr_init, flt_bbr, [('out_matrix_file', 'in_matrix_file')]),
        (flt_bbr, invt_bbr, [('out_matrix_file', 'in_file')]),
        (invt_bbr, segm_xfm, [('out_file', 'in_matrix_file')]),
        (inputnode, segm_xfm, [('epi_mean', 'reference')]),
        (segm_xfm, outputnode, [('out_file', 'epi_parc')]),
        (flt_bbr, outputnode, [('out_file', 'epi_mni')]),

    ])
    return workflow


def spikes_mask(in_file, in_mask, out_file=None):
    import os.path as op
    import nibabel as nb
    import numpy as np
    from nilearn.image import mean_img
    from nilearn.plotting import plot_roi
    from scipy import ndimage as nd

    if out_file is None:
        fname, ext = op.splitext(op.basename(in_file))
        if ext == '.gz':
            fname, ext2 = op.splitext(fname)
            ext = ext2 + ext
        out_file = op.abspath('{}_spmask{}'.format(fname, ext))
        out_plot = op.abspath('{}_spmask.pdf'.format(fname))

    in_4d_nii = nb.load(in_file)
    func = in_4d_nii.get_data()
    orientation = nb.aff2axcodes(in_4d_nii.affine)

    mask_data = nb.load(in_mask).get_data()
    a = np.where(mask_data != 0)
    bbox = np.max(a[0]) - np.min(a[0]), np.max(a[1]) - \
        np.min(a[1]), np.max(a[2]) - np.min(a[2])
    longest_axis = np.argmax(bbox)

    # Input here is a binarized and intersected mask data from previous section
    dil_mask = nd.binary_dilation(
        mask_data, iterations=int(mask_data.shape[longest_axis]/9))

    rep = list(mask_data.shape)
    rep[longest_axis] = -1
    new_mask_2d = dil_mask.max(axis=longest_axis).reshape(rep)

    rep = [1, 1, 1]
    rep[longest_axis] = mask_data.shape[longest_axis]
    new_mask_3d = np.logical_not(np.tile(new_mask_2d, rep))

    if orientation[0] in ['L', 'R']:
        new_mask_3d[0, :, :] = True
        new_mask_3d[-1, :, :] = True
    else:
        new_mask_3d[:, 0, :] = True
        new_mask_3d[:, -1, :] = True

    mask_nii = nb.Nifti1Image(new_mask_3d.astype(np.uint8), in_4d_nii.get_affine(),
                              in_4d_nii.get_header())
    mask_nii.to_filename(out_file)

    plot_roi(mask_nii, mean_img(in_4d_nii), output_file=out_plot)
    return out_file, out_plot


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


def _big_plot(in_func, in_mask, in_segm, in_spikes, in_spikes_bg,
              in_spikes_fft, fd, dvars, out_file=None):
    import os.path as op
    import numpy as np
    from mriqc.viz.fmriplots import fMRIPlot
    if out_file is None:
        fname, ext = op.splitext(op.basename(in_func))
        if ext == '.gz':
            fname, _ = op.splitext(fname)
        out_file = op.abspath('{}_fmriplot.pdf'.format(fname))

    myplot = fMRIPlot(in_func, in_mask, in_segm)
    myplot.add_spikes(np.loadtxt(in_spikes), title='Axial slice homogeneity (brain mask)')
    myplot.add_spikes(np.loadtxt(in_spikes_bg),
                      zscored=False, title='Axial slice homogeneity (air mask)')
    myplot.add_spikes(np.loadtxt(in_spikes_fft),
                      zscored=False, title='Energy of spectrum (axial slice -wise)')
    myplot.add_confounds([0] + np.loadtxt(fd).tolist(), 'FD')
    myplot.add_confounds([0] + np.loadtxt(dvars).tolist(), 'DVARS')
    myplot.plot()
    myplot.fig.savefig(out_file, dpi=300, bbox_inches='tight')
    myplot.fig.clf()
    return out_file
