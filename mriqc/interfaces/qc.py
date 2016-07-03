#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# @Author: oesteban
# @Date:   2016-01-05 11:29:40
# @Email:  code@oscaresteban.es
# @Last modified by:   oesteban
# @Last Modified time: 2016-05-04 15:30:56
""" Nipype interfaces to quality control measures """

import numpy as np
import nibabel as nb

from mriqc.qc.anatomical import (snr, cnr, fber, efc, art_qi1, art_qi2,
                                 volume_fraction, rpve, summary_stats, cjv)
from mriqc.qc.functional import (gsr, gcor, summary_fd)

from nipype.interfaces.base import (BaseInterface, traits, TraitedSpec, File,
                                    InputMultiPath, BaseInterfaceInputSpec)
from nipype import logging
IFLOGGER = logging.getLogger('interface')

class StructuralQCInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True, desc='file to be plotted')
    in_noinu = File(exists=True, mandatory=True, desc='image after INU correction')
    in_segm = File(exists=True, mandatory=True, desc='segmentation file from FSL FAST')
    in_bias = File(exists=True, mandatory=True, desc='bias file')
    air_msk = File(exists=True, mandatory=True, desc='air mask')
    artifact_msk = File(exists=True, mandatory=True, desc='air mask')
    in_pvms = InputMultiPath(File(exists=True), mandatory=True,
                             desc='partial volume maps from FSL FAST')
    in_tpms = InputMultiPath(File(), desc='tissue probability maps from FSL FAST')


class StructuralQCOutputSpec(TraitedSpec):
    summary = traits.Dict(desc='summary statistics per tissue')
    icvs = traits.Dict(desc='intracranial volume (ICV) fractions')
    rpve = traits.Dict(desc='partial volume fractions')
    size = traits.Dict(desc='image sizes')
    spacing = traits.Dict(desc='image sizes')
    inu = traits.Dict(desc='summary statistics of the bias field')
    snr = traits.Dict
    cnr = traits.Float
    fber = traits.Float
    efc = traits.Float
    qi1 = traits.Float
    qi2 = traits.Float
    cjv = traits.Float
    out_qc = traits.Dict(desc='output flattened dictionary with all measures')


class StructuralQC(BaseInterface):
    """
    Computes anatomical :abbr:`QC (Quality Control)` measures on the
    structural image given as input

    """
    input_spec = StructuralQCInputSpec
    output_spec = StructuralQCOutputSpec

    def __init__(self, **inputs):
        self._results = {}
        super(StructuralQC, self).__init__(**inputs)

    def _list_outputs(self):
        return self._results

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

        # SNR
        snrvals = []
        self._results['snr'] = {}
        for tlabel in ['csf', 'wm', 'gm']:
            snrvals.append(snr(inudata, segdata, airdata, fglabel=tlabel,
                               erode=erode))
            self._results['snr'][tlabel] = snrvals[-1]
        self._results['snr']['total'] = float(np.mean(snrvals))

        # CNR
        self._results['cnr'] = cnr(inudata, segdata)

        # FBER
        self._results['fber'] = fber(inudata, segdata, airdata)

        # EFC
        self._results['efc'] = efc(inudata)

        # Artifacts
        self._results['qi1'] = art_qi1(airdata, artdata)
        self._results['qi2'] = art_qi2(imdata, airdata, artdata)

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
        mean, stdv, p95, p05 = summary_stats(imdata, pvmdata)
        self._results['summary'] = {'mean': mean, 'stdv': stdv,
                                    'p95': p95, 'p05': p05}

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


        # Flatten the dictionary
        self._results['out_qc'] = _flatten_dict(self._results)
        return runtime


class FunctionalQCInputSpec(BaseInterfaceInputSpec):
    in_epi = File(exists=True, mandatory=True, desc='input EPI file')
    in_hmc = File(exists=True, mandatory=True, desc='input motion corrected file')
    in_tsnr = File(exists=True, mandatory=True, desc='input tSNR volume')
    in_mask = File(exists=True, mandatory=True, desc='input mask')
    direction = traits.Enum('all', 'x', 'y', '-x', '-y', usedefault=True,
                            desc='direction for GSR computation')
    fd_movpar = File(exists=True, mandatory=True, desc='motion parameters for FD computation')
    fd_thres = traits.Float(1., usedefault=True, desc='motion threshold for FD computation')
    in_dvars = File(exists=True, mandatory=True, desc='input file containing DVARS')



class FunctionalQCOutputSpec(TraitedSpec):
    fber = traits.Float
    efc = traits.Float
    snr = traits.Float
    gsr = traits.Dict
    m_tsnr = traits.Float
    dvars = traits.Float
    gcor = traits.Float
    fd_stats = traits.Dict
    size = traits.Dict
    spacing = traits.Dict
    summary = traits.Dict

    out_qc = traits.Dict(desc='output flattened dictionary with all measures')


class FunctionalQC(BaseInterface):
    """
    Computes anatomical :abbr:`QC (Quality Control)` measures on the
    structural image given as input

    """
    input_spec = FunctionalQCInputSpec
    output_spec = FunctionalQCOutputSpec

    def __init__(self, **inputs):
        self._results = {}
        super(FunctionalQC, self).__init__(**inputs)

    def _list_outputs(self):
        return self._results

    def _run_interface(self, runtime):
        # Get the mean EPI data and get it ready
        epinii = nb.load(self.inputs.in_epi)
        epidata = np.nan_to_num(epinii.get_data())
        epidata = epidata.astype(np.float32)
        epidata[epidata < 0] = 0

        # Get EPI data (with mc done) and get it ready
        hmcnii = nb.load(self.inputs.in_hmc)
        hmcdata = np.nan_to_num(hmcnii.get_data())
        hmcdata = hmcdata.astype(np.float32)
        hmcdata[hmcdata < 0] = 0

        # Get EPI data (with mc done) and get it ready
        msknii = nb.load(self.inputs.in_mask)
        mskdata = np.nan_to_num(msknii.get_data())
        mskdata = mskdata.astype(np.uint8)
        mskdata[mskdata < 0] = 0
        mskdata[mskdata > 0] = 1

        # SNR
        self._results['snr'] = float(snr(epidata, mskdata, fglabel=1))
        # FBER
        self._results['fber'] = fber(epidata, mskdata)
        # EFC
        self._results['efc'] = efc(epidata)
        # GSR
        self._results['gsr'] = {}
        if self.inputs.direction == 'all':
            epidir = ['x', 'y']
        else:
            epidir = [self.inputs.direction]

        for axis in epidir:
            self._results['gsr'][axis] = gsr(epidata, mskdata, direction=axis)

        # Summary stats
        mean, stdv, p95, p05 = summary_stats(epidata, mskdata)
        self._results['summary'] = {'mean': mean,
                                    'stdv': stdv,
                                    'p95': p95,
                                    'p05': p05}

        # DVARS
        self._results['dvars'] = float(np.loadtxt(
            self.inputs.in_dvars).mean())

        # tSNR
        tsnr_data = nb.load(self.inputs.in_tsnr).get_data()
        self._results['m_tsnr'] = float(np.median(tsnr_data[mskdata > 0]))

        # GCOR
        self._results['gcor'] = gcor(hmcdata, mskdata)

        # FD
        self._results['fd_stats'] = summary_fd(self.inputs.fd_movpar)

        # Image specs
        self._results['size'] = {'x': int(hmcdata.shape[0]),
                                 'y': int(hmcdata.shape[1]),
                                 'z': int(hmcdata.shape[2])}
        self._results['spacing'] = {
            i: float(v) for i, v in zip(['x', 'y', 'z'],
                                        hmcnii.get_header().get_zooms()[:3])}

        try:
            self._results['size']['t'] = int(hmcdata.shape[3])
        except IndexError:
            pass

        try:
            self._results['spacing']['tr'] = float(hmcnii.get_header().get_zooms()[3])
        except IndexError:
            pass

        self._results['out_qc'] = _flatten_dict(self._results)
        return runtime

def _flatten_dict(indict):
    out_qc = {}
    for k, value in list(indict.items()):
        if not isinstance(value, dict):
            out_qc[k] = value
        else:
            for subk, subval in list(value.items()):
                if not isinstance(subval, dict):
                    out_qc['%s_%s' % (k, subk)] = subval
                else:
                    for ssubk, ssubval in list(subval.items()):
                        out_qc['%s_%s_%s' % (k, subk, ssubk)] = ssubval
    return out_qc
