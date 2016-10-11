#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# @Author: oesteban
# @Date:   2016-01-05 11:24:05
# @Email:  code@oscaresteban.es
# @Last modified by:   oesteban
# @Last Modified time: 2016-10-04 14:54:16
""" A QC workflow for anatomical MRI """
from __future__ import print_function, division, absolute_import, unicode_literals
from builtins import zip, range
import os
import os.path as op
from nipype.pipeline import engine as pe
from nipype.interfaces import io as nio
from nipype.interfaces import utility as niu
from nipype.interfaces import fsl
from nipype.interfaces import ants
from nipype.interfaces.afni import preprocess as afp
from nipype.interfaces.freesurfer import MRIConvert

from niworkflows.data import get_mni_template
from niworkflows.anat.skullstrip import afni_wf as skullstrip_wf
from niworkflows.anat.mni import RobustMNINormalization
from mriqc.workflows.utils import fwhm_dict
from mriqc.interfaces.qc import StructuralQC
from mriqc.interfaces.anatomical import ArtifactMask
from mriqc.interfaces.bids import ReadSidecarJSON
from mriqc.utils.misc import bids_getfile, bids_path, check_folder, reorient
from ..reports.utils import plot_anat_mosaic_helper


def anat_qc_workflow(name='MRIQC_Anat', settings=None):
    """
    One-subject-one-session-one-run pipeline to extract the NR-IQMs from
    anatomical images
    """
    if settings is None:
        settings = {}

    workflow = pe.Workflow(name=name)
    deriv_dir = check_folder(op.abspath(op.join(settings['output_dir'], 'derivatives')))

    # Define workflow, inputs and outputs
    inputnode = pe.Node(niu.IdentityInterface(
        fields=['bids_dir', 'subject_id', 'session_id',
                'run_id']), name='inputnode')
    outputnode = pe.Node(niu.IdentityInterface(fields=['out_json']), name='outputnode')


    # 0. Get data
    datasource = pe.Node(niu.Function(
        input_names=['bids_dir', 'data_type', 'subject_id', 'session_id', 'run_id'],
        output_names=['anatomical_scan'], function=bids_getfile), name='datasource')
    datasource.inputs.data_type = 'anat'

    meta = pe.Node(ReadSidecarJSON(), name='metadata')

    # 1a. Reorient anatomical image
    arw = pe.Node(niu.Function(input_names=['in_file'],
                               output_names=["out_file"],
                               function=reorient), name='Reorient')
    # 1b. Estimate bias
    n4itk = pe.Node(ants.N4BiasFieldCorrection(dimension=3, save_bias=True), name='Bias')
    # 2. Skull-stripping (afni)
    asw = skullstrip_wf()
    mask = pe.Node(fsl.ApplyMask(), name='MaskAnatomical')
    # 3. Head mask (including nasial-cerebelum mask)
    hmsk = headmsk_wf()
    # 4. Air mask (with and without artifacts)
    amw = airmsk_wf(settings=settings)

    # Brain tissue segmentation
    segment = pe.Node(fsl.FAST(
        img_type=1, segments=True, out_basename='segment'), name='segmentation')

    # AFNI check smoothing
    fwhm = pe.Node(afp.FWHMx(combine=True, detrend=True), name='smoothness')
    # fwhm.inputs.acf = True  # add when AFNI >= 16

    # Compute python-coded measures
    measures = pe.Node(StructuralQC(testing=settings.get('testing', False)),
                       'measures')

    # T1w mosaic plot
    plot_anat_mosaic_zoomed = pe.Node(niu.Function(input_names=["in_file",
                                                                'subject_id',
                                                                'session_id',
                                                                'run_id',
                                                                "out_name",
                                                                "title",
                                                                "bbox_mask_file"],
                                            output_names=["plot_file"],
                                            function=plot_anat_mosaic_helper),
                                            name="plot_anat_mosaic_zoomed"
                                            )
    plot_anat_mosaic_zoomed.inputs.out_name = "plot_anat_mosaic1_zoomed.pdf"
    plot_anat_mosaic_zoomed.inputs.title = 'T1w (zoomed) session: {session_id} run: {run_id}'

    plot_anat_mosaic_noise = pe.Node(niu.Function(input_names=["in_file",
                                                                'subject_id',
                                                                'session_id',
                                                                'run_id',
                                                                "out_name",
                                                                "title",
                                                                "only_plot_noise"],
                                            output_names=["plot_file"],
                                            function=plot_anat_mosaic_helper),
                                            name="plot_anat_mosaic_noise"
                                            )
    plot_anat_mosaic_noise.inputs.only_plot_noise = True
    plot_anat_mosaic_noise.inputs.out_name = "plot_anat_mosaic2_noise.pdf"
    plot_anat_mosaic_noise.inputs.title = 'T1w (noise) session: {session_id} run: {run_id}'

    # Link images that should be reported
    dsreport = pe.Node(nio.DataSink(
        base_directory=settings['report_dir'], parameterization=True), name='dsreport')
    dsreport.inputs.container = 'anat'
    dsreport.inputs.substitutions = [
        ('_data', ''),
        ('background_fit', 'plot_bgfit')
    ]
    dsreport.inputs.regexp_substitutions = [
        ('_u?(sub-[\\w\\d]*)\\.([\\w\\d_]*)(?:\\.([\\w\\d_-]*))+', '\\1_ses-\\2_\\3'),
        ('anatomical_bgplotsub-[^/.]*_dvars_std', 'plot_dvars'),
        ('sub-[^/.]*_T1w_out_calc_thresh', 'mask'),
    ]

    # Connect all nodes
    workflow.connect([
        (inputnode, datasource, [('bids_dir', 'bids_dir'),
                                 ('subject_id', 'subject_id'),
                                 ('session_id', 'session_id'),
                                 ('run_id', 'run_id')]),
        (datasource, arw, [('anatomical_scan', 'in_file')]),
        (datasource, meta, [('anatomical_scan', 'in_file')]),
        (arw, asw, [('out_file', 'inputnode.in_file')]),
        (arw, n4itk, [('out_file', 'input_image')]),
        # (asw, n4itk, [('outputnode.out_mask', 'mask_image')]),
        (n4itk, mask, [('output_image', 'in_file')]),
        (asw, mask, [('outputnode.out_mask', 'mask_file')]),
        (mask, segment, [('out_file', 'in_files')]),
        (n4itk, hmsk, [('output_image', 'inputnode.in_file')]),
        (segment, hmsk, [('tissue_class_map', 'inputnode.in_segm')]),
        (n4itk, measures, [('output_image', 'in_noinu')]),
        (arw, measures, [('out_file', 'in_file')]),
        (arw, fwhm, [('out_file', 'in_file')]),
        (asw, fwhm, [('outputnode.out_mask', 'mask')]),

        (arw, amw, [('out_file', 'inputnode.in_file')]),
        (n4itk, amw, [('output_image', 'inputnode.in_noinu')]),
        (asw, amw, [('outputnode.out_mask', 'inputnode.in_mask')]),
        (hmsk, amw, [('outputnode.out_file', 'inputnode.head_mask')]),

        (amw, measures, [('outputnode.out_file', 'air_msk')]),
        (amw, measures, [('outputnode.artifact_msk', 'artifact_msk')]),

        (segment, measures, [('tissue_class_map', 'in_segm'),
                             ('partial_volume_files', 'in_pvms')]),
        (n4itk, measures, [('bias_image', 'in_bias')]),

        (arw, plot_anat_mosaic_zoomed, [('out_file', 'in_file')]),
        (asw, plot_anat_mosaic_zoomed, [('outputnode.out_mask', 'bbox_mask_file')]),
        (inputnode, plot_anat_mosaic_zoomed, [('subject_id', 'subject_id'),
                                              ('session_id', 'session_id'),
                                              ('run_id', 'run_id')]),

        (arw, plot_anat_mosaic_noise, [('out_file', 'in_file')]),
        (inputnode, plot_anat_mosaic_noise, [('subject_id', 'subject_id'),
                                             ('session_id', 'session_id'),
                                             ('run_id', 'run_id')]),

        (measures, dsreport, [('out_noisefit', '@anat_noiseplot')]),
        (plot_anat_mosaic_zoomed, dsreport, [('plot_file', "@anat_mosaic_zoomed")]),
        (plot_anat_mosaic_noise, dsreport, [('plot_file', "@anat_mosaic_noise")]),
        (asw, dsreport, [('outputnode.out_mask', '@anat_t1_mask')])
    ])

    # Format name
    out_name = pe.Node(niu.Function(
        input_names=['subid', 'sesid', 'runid', 'prefix', 'out_path'], output_names=['out_file'],
        function=bids_path), name='FormatName')
    out_name.inputs.out_path = deriv_dir
    out_name.inputs.prefix = 'anat'

    # Save to JSON file
    jfs_if = nio.JSONFileSink()
    setattr(jfs_if, '_always_run', settings.get('force_run', False))
    datasink = pe.Node(jfs_if, name='datasink')
    datasink.inputs.qc_type = 'anat'

    workflow.connect([
        (inputnode, out_name, [('subject_id', 'subid'),
                               ('session_id', 'sesid'),
                               ('run_id', 'runid')]),
        (inputnode, datasink, [('subject_id', 'subject_id'),
                               ('session_id', 'session_id'),
                               ('run_id', 'run_id')]),
        (fwhm, datasink, [(('fwhm', fwhm_dict), 'fwhm')]),
        (measures, datasink, [('summary', 'summary'),
                              ('spacing', 'spacing'),
                              ('size', 'size'),
                              ('icvs', 'icvs'),
                              ('rpve', 'rpve'),
                              ('inu', 'inu'),
                              ('snr', 'snr'),
                              ('cnr', 'cnr'),
                              ('fber', 'fber'),
                              ('efc', 'efc'),
                              ('qi1', 'qi1'),
                              ('qi2', 'qi2'),
                              ('cjv', 'cjv'),
                              ('wm2max', 'wm2max')]),
        (out_name, datasink, [('out_file', 'out_file')]),
        (meta, datasink, [('out_dict', 'metadata')]),
        (datasink, outputnode, [('out_file', 'out_file')])
    ])
    return workflow


def headmsk_wf(name='HeadMaskWorkflow', use_bet=True):
    """Computes a head mask as in [Mortamet2009]_."""

    has_dipy = False
    try:
        from dipy.denoise import nlmeans
        from nipype.interfaces.dipy import Denoise
        has_dipy = True
    except ImportError:
        pass

    workflow = pe.Workflow(name=name)
    inputnode = pe.Node(niu.IdentityInterface(fields=['in_file', 'in_segm']),
                        name='inputnode')
    outputnode = pe.Node(niu.IdentityInterface(fields=['out_file']), name='outputnode')


    enhance = pe.Node(niu.Function(
        input_names=['in_file'], output_names=['out_file'], function=_enhance), name='Enhance')


    if use_bet or not has_dipy:
        # Alternative for when dipy is not installed
        bet = pe.Node(fsl.BET(surfaces=True), name='fsl_bet')
        workflow.connect([
            (inputnode, enhance, [('in_file', 'in_file')]),
            (enhance, bet, [('out_file', 'in_file')]),
            (bet, outputnode, [('outskin_mask_file', 'out_file')])
        ])
        return workflow

    estsnr = pe.Node(niu.Function(
        input_names=['in_file', 'seg_file'], output_names=['out_snr'],
        function=_estimate_snr), name='EstimateSNR')
    denoise = pe.Node(Denoise(), name='Denoise')
    gradient = pe.Node(niu.Function(
        input_names=['in_file', 'snr'], output_names=['out_file'], function=image_gradient), name='Grad')
    thresh = pe.Node(niu.Function(
        input_names=['in_file', 'in_segm'], output_names=['out_file'], function=gradient_threshold),
                     name='GradientThreshold')

    workflow.connect([
        (inputnode, estsnr, [('in_file', 'in_file'),
                             ('in_segm', 'seg_file')]),
        (estsnr, denoise, [('out_snr', 'snr')]),
        (inputnode, enhance, [('in_file', 'in_file')]),
        (enhance, denoise, [('out_file', 'in_file')]),
        (estsnr, gradient, [('out_snr', 'snr')]),
        (denoise, gradient, [('out_file', 'in_file')]),
        (inputnode, thresh, [('in_segm', 'in_segm')]),
        (gradient, thresh, [('out_file', 'in_file')]),
        (thresh, outputnode, [('out_file', 'out_file')])
    ])

    return workflow


def airmsk_wf(name='AirMaskWorkflow', settings=None):
    """Implements the Step 1 of [Mortamet2009]_."""
    import pkg_resources as pkgr
    workflow = pe.Workflow(name=name)

    if settings is None:
        settings = {}

    inputnode = pe.Node(niu.IdentityInterface(
        fields=['in_file', 'in_noinu', 'in_mask', 'head_mask']), name='inputnode')
    outputnode = pe.Node(niu.IdentityInterface(fields=['out_file', 'artifact_msk']),
                         name='outputnode')

    # Spatial normalization, using ANTs
    norm = pe.Node(RobustMNINormalization(num_threads=settings.get('ants_nthreads', 4)),
                   name='normalize')
    norm.inputs.testing = settings.get('testing', False)

    invt = pe.Node(ants.ApplyTransforms(
        dimension=3, default_value=1, interpolation='NearestNeighbor'), name='invert_xfm')
    invt.inputs.input_image = op.join(get_mni_template(), 'MNI152_T1_1mm_brain_bottom.nii.gz')

    qi1 = pe.Node(ArtifactMask(), name='ArtifactMask')

    workflow.connect([
        (inputnode, qi1, [('in_file', 'in_file')]),
        (inputnode, norm, [('in_noinu', 'moving_image'),
                           ('in_mask', 'moving_mask')]),
        (norm, invt, [('reverse_transforms', 'transforms'),
                      ('reverse_invert_flags', 'invert_transform_flags')]),
        (inputnode, invt, [('in_mask', 'reference_image')]),
        (inputnode, qi1, [('head_mask', 'head_mask')]),
        (invt, qi1, [('output_image', 'nasion_post_mask')]),
        (qi1, outputnode, [('out_air_msk', 'out_file'),
                           ('out_art_msk', 'artifact_msk')])
    ])
    return workflow

def _estimate_snr(in_file, seg_file):
    import nibabel as nb
    from mriqc.qc.anatomical import snr
    out_snr = snr(nb.load(in_file).get_data(), nb.load(seg_file).get_data(),
                  fglabel='wm')
    return out_snr

def _enhance(in_file, out_file=None):
    import os.path as op
    import numpy as np
    import nibabel as nb

    if out_file is None:
        fname, ext = op.splitext(op.basename(in_file))
        if ext == '.gz':
            fname, ext2 = op.splitext(fname)
            ext = ext2 + ext
        out_file = op.abspath('{}_enhanced{}'.format(fname, ext))

    imnii = nb.load(in_file)
    data = imnii.get_data().astype(np.float32)  # pylint: disable=no-member
    range_max = np.percentile(data[data > 0], 99.98)
    range_min = np.median(data[data > 0])

    # Resample signal excess pixels
    excess = np.where(data > range_max)
    data[excess] = 0
    data[excess] = np.random.choice(data[data > range_min], size=len(excess[0]))

    nb.Nifti1Image(data, imnii.get_affine(), imnii.get_header()).to_filename(
        out_file)

    return out_file

def image_gradient(in_file, snr, out_file=None):
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
        out_file = op.abspath('{}_grad{}'.format(fname, ext))

    imnii = nb.load(in_file)
    data = imnii.get_data().astype(np.float32)  # pylint: disable=no-member
    datamax = np.percentile(data.reshape(-1), 99.5)
    data *= 100 / datamax
    grad = gradient(data, 3.0)
    gradmax = np.percentile(grad.reshape(-1), 99.5)
    grad *= 100.
    grad /= gradmax

    nb.Nifti1Image(grad, imnii.get_affine(), imnii.get_header()).to_filename(out_file)
    return out_file

def gradient_threshold(in_file, in_segm, thresh=1.0, out_file=None):
    """ Compute a threshold from the histogram of the magnitude gradient image """
    import os.path as op
    import numpy as np
    import nibabel as nb
    from scipy import ndimage as sim

    struc = sim.iterate_structure(sim.generate_binary_structure(3, 2), 2)

    if out_file is None:
        fname, ext = op.splitext(op.basename(in_file))
        if ext == '.gz':
            fname, ext2 = op.splitext(fname)
            ext = ext2 + ext
        out_file = op.abspath('{}_gradmask{}'.format(fname, ext))

    imnii = nb.load(in_file)

    hdr = imnii.get_header().copy()
    hdr.set_data_dtype(np.uint8)  # pylint: disable=no-member

    data = imnii.get_data().astype(np.float32)

    mask = np.zeros_like(data, dtype=np.uint8)  # pylint: disable=no-member
    mask[data > 15.] = 1

    segdata = nb.load(in_segm).get_data().astype(np.uint8)
    segdata[segdata > 0] = 1
    segdata = sim.binary_dilation(segdata, struc, iterations=2, border_value=1).astype(np.uint8)  # pylint: disable=no-member
    mask[segdata > 0] = 1

    mask = sim.binary_closing(mask, struc, iterations=2).astype(np.uint8)  # pylint: disable=no-member
    # Remove small objects
    label_im, nb_labels = sim.label(mask)
    artmsk = np.zeros_like(mask)
    if nb_labels > 2:
        sizes = sim.sum(mask, label_im, list(range(nb_labels + 1)))
        ordered = list(reversed(sorted(zip(sizes, list(range(nb_labels + 1))))))
        for _, label in ordered[2:]:
            mask[label_im == label] = 0
            artmsk[label_im == label] = 1

    mask = sim.binary_fill_holes(mask, struc).astype(np.uint8)  # pylint: disable=no-member

    nb.Nifti1Image(mask, imnii.get_affine(), hdr).to_filename(out_file)
    return out_file

def _bgplot(in_file, base_directory):
    from nipype.interfaces.io import DataSink
    if not in_file:
        return ''

    ds = DataSink(base_directory=base_directory, parameterization=False)
    setattr(ds.inputs, '@bg_fitting', in_file)
    return ds.run().outputs.out_file
