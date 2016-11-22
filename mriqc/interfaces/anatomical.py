#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# @Author: oesteban
# @Date:   2016-01-05 11:29:40
# @Email:  code@oscaresteban.es
# @Last modified by:   oesteban
# @Last Modified time: 2016-11-21 18:59:13
""" Nipype interfaces to support anatomical workflow """
from __future__ import print_function, division, absolute_import, unicode_literals
import os.path as op
import numpy as np
import nibabel as nb
import scipy.ndimage as nd
from builtins import zip

from nipype import logging
from nipype.interfaces.base import (traits, TraitedSpec, File,
                                    InputMultiPath, BaseInterfaceInputSpec)

from mriqc.utils.misc import _flatten_dict
from mriqc.interfaces.base import MRIQCBaseInterface
from mriqc.qc.anatomical import (snr, snr_dietrich, cnr, fber, efc, art_qi1,
                                 art_qi2, volume_fraction, rpve, summary_stats,
                                 cjv, wm2max)
IFLOGGER = logging.getLogger('interface')


class StructuralQCInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True, desc='file to be plotted')
    in_noinu = File(exists=True, mandatory=True, desc='image after INU correction')
    in_segm = File(exists=True, mandatory=True, desc='segmentation file from FSL FAST')
    in_bias = File(exists=True, mandatory=True, desc='bias file')
    head_msk = File(exists=True, mandatory=True, desc='head mask')
    air_msk = File(exists=True, mandatory=True, desc='air mask')
    artifact_msk = File(exists=True, mandatory=True, desc='air mask')
    in_pvms = InputMultiPath(File(exists=True), mandatory=True,
                             desc='partial volume maps from FSL FAST')
    in_tpms = InputMultiPath(File(), desc='tissue probability maps from FSL FAST')
    mni_tpms = InputMultiPath(File(), desc='tissue probability maps from FSL FAST')


class StructuralQCOutputSpec(TraitedSpec):
    summary = traits.Dict(desc='summary statistics per tissue')
    icvs = traits.Dict(desc='intracranial volume (ICV) fractions')
    rpve = traits.Dict(desc='partial volume fractions')
    size = traits.Dict(desc='image sizes')
    spacing = traits.Dict(desc='image sizes')
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


class StructuralQC(MRIQCBaseInterface):
    """
    Computes anatomical :abbr:`QC (Quality Control)` measures on the
    structural image given as input

    """
    input_spec = StructuralQCInputSpec
    output_spec = StructuralQCOutputSpec

    def _run_interface(self, runtime):  # pylint: disable=R0914
        imnii = nb.load(self.inputs.in_file)
        imdata = np.nan_to_num(imnii.get_data())
        erode = np.all(np.array(imnii.get_header().get_zooms()[:3],
                                dtype=np.float32) < 1.2)

        # Cast to float32
        imdata = imdata.astype(np.float32)

        # Remove negative values
        imdata[imdata < 0] = 0

        # Load image corrected for INU
        inudata = np.nan_to_num(nb.load(self.inputs.in_noinu).get_data())
        inudata[inudata < 0] = 0

        segnii = nb.load(self.inputs.in_segm)
        segdata = segnii.get_data().astype(np.uint8)

        airdata = nb.load(self.inputs.air_msk).get_data().astype(np.uint8)
        artdata = nb.load(self.inputs.artifact_msk).get_data().astype(np.uint8)
        headdata = nb.load(self.inputs.head_msk).get_data().astype(np.uint8)

        # SNR
        snrvals = []
        self._results['snr'] = {}
        for tlabel in ['csf', 'wm', 'gm']:
            snrvals.append(snr(inudata, segdata, fglabel=tlabel, erode=erode))
            self._results['snr'][tlabel] = snrvals[-1]
        self._results['snr']['total'] = float(np.mean(snrvals))

        snrvals = []
        self._results['snrd'] = {
            tlabel: snr_dietrich(inudata, segdata, airdata, fglabel=tlabel, erode=erode)
            for tlabel in ['csf', 'wm', 'gm']}
        self._results['snrd']['total'] = float(
            np.mean([val for _, val in list(self._results['snrd'].items())]))

        # CNR
        self._results['cnr'] = cnr(inudata, segdata)

        # FBER
        self._results['fber'] = fber(inudata, headdata)

        # EFC
        self._results['efc'] = efc(inudata)

        # M2WM
        self._results['wm2max'] = wm2max(imdata, segdata)

        # Artifacts
        self._results['qi_1'] = art_qi1(airdata, artdata)

        # CJV
        self._results['cjv'] = cjv(inudata, seg=segdata)

        pvmdata = []
        for fname in self.inputs.in_pvms:
            pvmdata.append(nb.load(fname).get_data().astype(np.float32))

        # ICVs
        self._results['icvs'] = volume_fraction(pvmdata)

        # RPVE
        self._results['rpve'] = rpve(pvmdata, segdata)

        # Summary stats
        self._results['summary'] = summary_stats(imdata, pvmdata, airdata)

        # Image specs
        self._results['size'] = {'x': int(imdata.shape[0]),
                                 'y': int(imdata.shape[1]),
                                 'z': int(imdata.shape[2])}
        self._results['spacing'] = {
            i: float(v) for i, v in zip(
                ['x', 'y', 'z'], imnii.get_header().get_zooms()[:3])}

        try:
            self._results['size']['t'] = int(imdata.shape[3])
        except IndexError:
            pass

        try:
            self._results['spacing']['tr'] = float(imnii.get_header().get_zooms()[3])
        except IndexError:
            pass

        # Bias
        bias = nb.load(self.inputs.in_bias).get_data()[segdata > 0]
        self._results['inu'] = {
            'range': float(np.abs(np.percentile(bias, 95.) - np.percentile(bias, 5.))),
            'med': float(np.median(bias))}  #pylint: disable=E1101

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
    nasion_post_mask = File(exists=True, mandatory=True,
                            desc='nasion to posterior of cerebellum mask')


class ArtifactMaskOutputSpec(TraitedSpec):
    out_art_msk = File(exists=True, desc='output artifacts mask')
    out_air_msk = File(exists=True, desc='output artifacts mask, without artifacts')


class ArtifactMask(MRIQCBaseInterface):
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

        # Run the artifact detection
        qi1_img = artifact_mask(imdata, airdata, dist)

        fname, ext = op.splitext(op.basename(self.inputs.in_file))
        if ext == '.gz':
            fname, ext2 = op.splitext(fname)
            ext = ext2 + ext

        self._results['out_art_msk'] = op.abspath('{}_artifacts{}'.format(fname, ext))
        self._results['out_air_msk'] = op.abspath('{}_noart-air{}'.format(fname, ext))

        hdr = imnii.get_header().copy()
        hdr.set_data_dtype(np.uint8)
        nb.Nifti1Image(qi1_img, imnii.get_affine(), hdr).to_filename(
            self._results['out_art_msk'])

        airdata[qi1_img > 0] = 0
        nb.Nifti1Image(airdata, imnii.get_affine(), hdr).to_filename(
            self._results['out_air_msk'])
        return runtime


class ComputeQI2InputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True, desc='File to be plotted')
    air_msk = File(exists=True, mandatory=True, desc='air (without artifacts) mask')
    erodemsk = traits.Bool(True, usedefault=True, desc='erode mask')
    ncoils = traits.Int(12, usedefault=True, desc='number of coils')

class ComputeQI2OutputSpec(TraitedSpec):
    qi2 = traits.Float(desc='computed QI2 value')
    out_file = File(desc='output plot: noise fit')


class ComputeQI2(MRIQCBaseInterface):
    """
    Computes the artifact mask using the method described in [Mortamet2009]_.
    """
    input_spec = ComputeQI2InputSpec
    output_spec = ComputeQI2OutputSpec

    def _run_interface(self, runtime):
        imdata = nb.load(self.inputs.in_file).get_data()
        airdata = nb.load(self.inputs.air_msk).get_data()
        qi2, out_file = art_qi2(imdata, airdata, ncoils=self.inputs.ncoils,
                                erodemask=self.inputs.erodemsk)
        self._results['qi2'] = qi2
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
        overlaps.append(float(num/den))
    return overlaps
