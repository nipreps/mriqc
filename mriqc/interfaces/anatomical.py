#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# @Author: oesteban
# @Date:   2016-01-05 11:29:40
# @Email:  code@oscaresteban.es
""" Nipype interfaces to support anatomical workflow """
import os.path as op
import numpy as np
import nibabel as nb
from math import sqrt
import scipy.ndimage as nd
from builtins import zip

from nipype import logging
from nipype.utils.filemanip import fname_presuffix
from nipype.interfaces.base import (
    traits, TraitedSpec, File, isdefined, InputMultiPath, BaseInterfaceInputSpec,
    SimpleInterface
)

from ..utils.misc import _flatten_dict
from ..qc.anatomical import (snr, snr_dietrich, cnr, fber, efc, art_qi1,
                             art_qi2, volume_fraction, rpve, summary_stats,
                             cjv, wm2max)
IFLOGGER = logging.getLogger('nipype.interface')


class StructuralQCInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True, desc='file to be plotted')
    in_noinu = File(exists=True, mandatory=True, desc='image after INU correction')
    in_segm = File(exists=True, mandatory=True, desc='segmentation file from FSL FAST')
    in_bias = File(exists=True, mandatory=True, desc='bias file')
    head_msk = File(exists=True, mandatory=True, desc='head mask')
    air_msk = File(exists=True, mandatory=True, desc='air mask')
    rot_msk = File(exists=True, mandatory=True, desc='rotation mask')
    artifact_msk = File(exists=True, mandatory=True, desc='air mask')
    in_pvms = InputMultiPath(File(exists=True), mandatory=True,
                             desc='partial volume maps from FSL FAST')
    in_tpms = InputMultiPath(File(), desc='tissue probability maps from FSL FAST')
    mni_tpms = InputMultiPath(File(), desc='tissue probability maps from FSL FAST')
    in_fwhm = traits.List(traits.Float, mandatory=True,
                          desc='smoothness estimated with AFNI')


class StructuralQCOutputSpec(TraitedSpec):
    summary = traits.Dict(desc='summary statistics per tissue')
    icvs = traits.Dict(desc='intracranial volume (ICV) fractions')
    rpve = traits.Dict(desc='partial volume fractions')
    size = traits.Dict(desc='image sizes')
    spacing = traits.Dict(desc='image sizes')
    fwhm = traits.Dict(desc='full width half-maximum measure')
    inu = traits.Dict(desc='summary statistics of the bias field')
    snr = traits.Dict
    snrd = traits.Dict
    cnr = traits.Float
    fber = traits.Float
    efc = traits.Float
    qi_1 = traits.Float
    wm2max = traits.Float
    cjv = traits.Float
    out_qc = traits.Dict(desc='output flattened dictionary with all measures')
    out_noisefit = File(exists=True, desc='plot of background noise and chi fitting')
    tpm_overlap = traits.Dict


class StructuralQC(SimpleInterface):
    """
    Computes anatomical :abbr:`QC (Quality Control)` measures on the
    structural image given as input

    """
    input_spec = StructuralQCInputSpec
    output_spec = StructuralQCOutputSpec

    def _run_interface(self, runtime):  # pylint: disable=R0914,E1101
        imnii = nb.load(self.inputs.in_noinu)
        erode = np.all(np.array(imnii.header.get_zooms()[:3],
                                dtype=np.float32) < 1.9)

        # Load image corrected for INU
        inudata = np.nan_to_num(imnii.get_data())
        inudata[inudata < 0] = 0

        # Load binary segmentation from FSL FAST
        segnii = nb.load(self.inputs.in_segm)
        segdata = segnii.get_data().astype(np.uint8)

        # Load air, artifacts and head masks
        airdata = nb.load(self.inputs.air_msk).get_data().astype(np.uint8)
        artdata = nb.load(self.inputs.artifact_msk).get_data().astype(np.uint8)
        headdata = nb.load(self.inputs.head_msk).get_data().astype(np.uint8)
        rotdata = nb.load(self.inputs.rot_msk).get_data().astype(np.uint8)

        # Load Partial Volume Maps (pvms) from FSL FAST
        pvmdata = []
        for fname in self.inputs.in_pvms:
            pvmdata.append(nb.load(fname).get_data().astype(np.float32))

        # Summary stats
        stats = summary_stats(inudata, pvmdata, airdata, erode=erode)
        self._results['summary'] = stats

        # SNR
        snrvals = []
        self._results['snr'] = {}
        for tlabel in ['csf', 'wm', 'gm']:
            snrvals.append(snr(stats[tlabel]['median'], stats[tlabel]['stdv'], stats[tlabel]['n']))
            self._results['snr'][tlabel] = snrvals[-1]
        self._results['snr']['total'] = float(np.mean(snrvals))

        snrvals = []
        self._results['snrd'] = {
            tlabel: snr_dietrich(stats[tlabel]['median'], stats['bg']['mad'])
            for tlabel in ['csf', 'wm', 'gm']}
        self._results['snrd']['total'] = float(
            np.mean([val for _, val in list(self._results['snrd'].items())]))

        # CNR
        self._results['cnr'] = cnr(
            stats['wm']['median'], stats['gm']['median'],
            sqrt(sum(stats[k]['stdv'] ** 2 for k in ['bg', 'gm', 'wm']))
        )

        # FBER
        self._results['fber'] = fber(inudata, headdata, rotdata)

        # EFC
        self._results['efc'] = efc(inudata, rotdata)

        # M2WM
        self._results['wm2max'] = wm2max(inudata, stats['wm']['median'])

        # Artifacts
        self._results['qi_1'] = art_qi1(airdata, artdata)

        # CJV
        self._results['cjv'] = cjv(
            # mu_wm, mu_gm, sigma_wm, sigma_gm
            stats['wm']['median'],
            stats['gm']['median'],
            stats['wm']['mad'],
            stats['gm']['mad']
        )

        # FWHM
        fwhm = np.array(self.inputs.in_fwhm[:3]) / np.array(imnii.header.get_zooms()[:3])
        self._results['fwhm'] = {
            'x': float(fwhm[0]), 'y': float(fwhm[1]), 'z': float(fwhm[2]),
            'avg': float(np.average(fwhm))}

        # ICVs
        self._results['icvs'] = volume_fraction(pvmdata)

        # RPVE
        self._results['rpve'] = rpve(pvmdata, segdata)

        # Image specs
        self._results['size'] = {'x': int(inudata.shape[0]),
                                 'y': int(inudata.shape[1]),
                                 'z': int(inudata.shape[2])}
        self._results['spacing'] = {
            i: float(v) for i, v in zip(
                ['x', 'y', 'z'], imnii.header.get_zooms()[:3])}

        try:
            self._results['size']['t'] = int(inudata.shape[3])
        except IndexError:
            pass

        try:
            self._results['spacing']['tr'] = float(imnii.header.get_zooms()[3])
        except IndexError:
            pass

        # Bias
        bias = nb.load(self.inputs.in_bias).get_data()[segdata > 0]
        self._results['inu'] = {
            'range': float(np.abs(np.percentile(bias, 95.) - np.percentile(bias, 5.))),
            'med': float(np.median(bias))}  # pylint: disable=E1101

        mni_tpms = [nb.load(tpm).get_data() for tpm in self.inputs.mni_tpms]
        in_tpms = [nb.load(tpm).get_data() for tpm in self.inputs.in_pvms]
        overlap = fuzzy_jaccard(in_tpms, mni_tpms)
        self._results['tpm_overlap'] = {
            'csf': overlap[0],
            'gm': overlap[1],
            'wm': overlap[2]
        }

        # Flatten the dictionary
        self._results['out_qc'] = _flatten_dict(self._results)
        return runtime


class ArtifactMaskInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True, desc='File to be plotted')
    head_mask = File(exists=True, mandatory=True, desc='head mask')
    rot_mask = File(exists=True, desc='a rotation mask')
    nasion_post_mask = File(exists=True, mandatory=True,
                            desc='nasion to posterior of cerebellum mask')


class ArtifactMaskOutputSpec(TraitedSpec):
    out_hat_msk = File(exists=True, desc='output "hat" mask')
    out_art_msk = File(exists=True, desc='output artifacts mask')
    out_air_msk = File(exists=True, desc='output "hat" mask, without artifacts')


class ArtifactMask(SimpleInterface):
    """
    Computes the artifact mask using the method described in [Mortamet2009]_.
    """
    input_spec = ArtifactMaskInputSpec
    output_spec = ArtifactMaskOutputSpec

    def _run_interface(self, runtime):
        imnii = nb.load(self.inputs.in_file)
        imdata = np.nan_to_num(imnii.get_data().astype(np.float32))

        # Remove negative values
        imdata[imdata < 0] = 0

        hmdata = nb.load(self.inputs.head_mask).get_data()
        npdata = nb.load(self.inputs.nasion_post_mask).get_data()

        # Invert head mask
        airdata = np.ones_like(hmdata, dtype=np.uint8)
        airdata[hmdata == 1] = 0

        # Calculate distance to border
        dist = nd.morphology.distance_transform_edt(airdata)

        # Apply nasion-to-posterior mask
        airdata[npdata == 1] = 0
        dist[npdata == 1] = 0
        dist /= dist.max()

        # Apply rotation mask (if supplied)
        if isdefined(self.inputs.rot_mask):
            rotmskdata = nb.load(self.inputs.rot_mask).get_data()
            airdata[rotmskdata == 1] = 0

        # Run the artifact detection
        qi1_img = artifact_mask(imdata, airdata, dist)

        fname, ext = op.splitext(op.basename(self.inputs.in_file))
        if ext == '.gz':
            fname, ext2 = op.splitext(fname)
            ext = ext2 + ext

        self._results['out_hat_msk'] = op.abspath('{}_hat{}'.format(fname, ext))
        self._results['out_art_msk'] = op.abspath('{}_art{}'.format(fname, ext))
        self._results['out_air_msk'] = op.abspath('{}_air{}'.format(fname, ext))

        hdr = imnii.header.copy()
        hdr.set_data_dtype(np.uint8)
        nb.Nifti1Image(qi1_img, imnii.affine, hdr).to_filename(
            self._results['out_art_msk'])

        nb.Nifti1Image(airdata, imnii.affine, hdr).to_filename(
            self._results['out_hat_msk'])

        airdata[qi1_img > 0] = 0
        nb.Nifti1Image(airdata, imnii.affine, hdr).to_filename(
            self._results['out_air_msk'])
        return runtime


class ComputeQI2InputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True, desc='File to be plotted')
    air_msk = File(exists=True, mandatory=True, desc='air (without artifacts) mask')


class ComputeQI2OutputSpec(TraitedSpec):
    qi2 = traits.Float(desc='computed QI2 value')
    out_file = File(desc='output plot: noise fit')


class ComputeQI2(SimpleInterface):
    """
    Computes the artifact mask using the method described in [Mortamet2009]_.
    """
    input_spec = ComputeQI2InputSpec
    output_spec = ComputeQI2OutputSpec

    def _run_interface(self, runtime):
        imdata = nb.load(self.inputs.in_file).get_data()
        airdata = nb.load(self.inputs.air_msk).get_data()
        qi2, out_file = art_qi2(imdata, airdata)
        self._results['qi2'] = qi2
        self._results['out_file'] = out_file
        return runtime


class HarmonizeInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True, desc='input data (after bias correction)')
    wm_mask = File(exists=True, mandatory=True, desc='white-matter mask')
    erodemsk = traits.Bool(True, usedefault=True, desc='erode mask')


class HarmonizeOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='input data (after intensity harmonization)')


class Harmonize(SimpleInterface):
    """
    Computes the artifact mask using the method described in [Mortamet2009]_.
    """
    input_spec = HarmonizeInputSpec
    output_spec = HarmonizeOutputSpec

    def _run_interface(self, runtime):

        in_file = nb.load(self.inputs.in_file)
        wm_mask = nb.load(self.inputs.wm_mask).get_data()
        wm_mask[wm_mask < 0.9] = 0
        wm_mask[wm_mask > 0] = 1
        wm_mask = wm_mask.astype(np.uint8)

        if self.inputs.erodemsk:
            # Create a structural element to be used in an opening operation.
            struc = nd.generate_binary_structure(3, 2)
            # Perform an opening operation on the background data.
            wm_mask = nd.binary_erosion(wm_mask, structure=struc).astype(np.uint8)

        data = in_file.get_data()
        data *= 1000.0 / np.median(data[wm_mask > 0])

        out_file = fname_presuffix(self.inputs.in_file,
                                   suffix='_harmonized', newpath='.')
        in_file.__class__(data, in_file.affine, in_file.header).to_filename(
            out_file)

        self._results['out_file'] = out_file

        return runtime


class RotationMaskInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True, desc='input data')


class RotationMaskOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='rotation mask (if any)')


class RotationMask(SimpleInterface):
    """
    Computes the artifact mask using the method described in [Mortamet2009]_.
    """
    input_spec = RotationMaskInputSpec
    output_spec = RotationMaskOutputSpec

    def _run_interface(self, runtime):
        in_file = nb.load(self.inputs.in_file)
        data = in_file.get_data()
        mask = data <= 0

        # Pad one pixel to control behavior on borders of binary_opening
        mask = np.pad(mask, pad_width=(1,), mode='constant', constant_values=1)

        # Remove noise
        struc = nd.generate_binary_structure(3, 2)
        mask = nd.binary_opening(mask, structure=struc).astype(
            np.uint8)

        # Remove small objects
        label_im, nb_labels = nd.label(mask)
        if nb_labels > 2:
            sizes = nd.sum(mask, label_im, list(range(nb_labels + 1)))
            ordered = list(reversed(sorted(zip(sizes, list(range(nb_labels + 1))))))
            for _, label in ordered[2:]:
                mask[label_im == label] = 0

        # Un-pad
        mask = mask[1:-1, 1:-1, 1:-1]

        # If mask is small, clean-up
        if mask.sum() < 500:
            mask = np.zeros_like(mask, dtype=np.uint8)

        out_img = in_file.__class__(mask, in_file.affine, in_file.header)
        out_img.header.set_data_dtype(np.uint8)

        out_file = fname_presuffix(self.inputs.in_file,
                                   suffix='_rotmask', newpath='.')
        out_img.to_filename(out_file)
        self._results['out_file'] = out_file
        return runtime


def artifact_mask(imdata, airdata, distance, zscore=10.):
    """Computes a mask of artifacts found in the air region"""
    from statsmodels.robust.scale import mad

    if not np.issubdtype(airdata.dtype, np.integer):
        airdata[airdata < .95] = 0
        airdata[airdata > 0.] = 1

    bg_img = imdata * airdata
    if np.sum((bg_img > 0).astype(np.uint8)) < 100:
        return np.zeros_like(airdata)

    # Find the background threshold (the most frequently occurring value
    # excluding 0)
    bg_location = np.median(bg_img[bg_img > 0])
    bg_spread = mad(bg_img[bg_img > 0])
    bg_img[bg_img > 0] -= bg_location
    bg_img[bg_img > 0] /= bg_spread

    # Apply this threshold to the background voxels to identify voxels
    # contributing artifacts.
    qi1_img = np.zeros_like(bg_img)
    qi1_img[bg_img > zscore] = 1
    qi1_img[distance < .10] = 0

    # Create a structural element to be used in an opening operation.
    struc = nd.generate_binary_structure(3, 1)
    qi1_img = nd.binary_opening(qi1_img, struc).astype(np.uint8)
    qi1_img[airdata <= 0] = 0

    return qi1_img


def fuzzy_jaccard(in_tpms, in_mni_tpms):
    overlaps = []
    for tpm, mni_tpm in zip(in_tpms, in_mni_tpms):
        tpm = tpm.reshape(-1)
        mni_tpm = mni_tpm.reshape(-1)

        num = np.min([tpm, mni_tpm], axis=0).sum()
        den = np.max([tpm, mni_tpm], axis=0).sum()
        overlaps.append(float(num / den))
    return overlaps
