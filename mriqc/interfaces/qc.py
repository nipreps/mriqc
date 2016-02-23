#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
# pylint: disable=no-member
#
# @Author: oesteban
# @Date:   2016-01-05 11:29:40
# @Email:  code@oscaresteban.es
# @Last modified by:   oesteban
# @Last Modified time: 2016-02-23 12:19:03
""" Nipype interfaces to quality control measures """

import nibabel as nb
import numpy as np
from ..qc.anatomical import (snr, cnr, fber, efc, artifacts,
                             volume_fraction, rpve, summary_stats)
from nipype.interfaces.base import (BaseInterface, traits, TraitedSpec, File,
                                    InputMultiPath, BaseInterfaceInputSpec)

from nipype import logging
IFLOGGER = logging.getLogger('interface')

class StructuralQCInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True, desc='File to be plotted')
    in_segm = File(exists=True, mandatory=True,
                   desc='segmentation file from FSL FAST')
    in_bias = File(exists=True, mandatory=True,
                   desc='bias file')
    in_pvms = InputMultiPath(File(exists=True), mandatory=True,
                             desc='partial volume maps from FSL FAST')
    in_tpms = InputMultiPath(File(), desc='tissue probability maps from FSL FAST')


class StructuralQCOutputSpec(TraitedSpec):
    mean = traits.Dict(traits.Float, desc='mean intensity value per tissue')
    stdv = traits.Dict(traits.Float, desc='standard deviation of intensity')
    icvs = traits.Dict(traits.Float, desc='intracranial volume (ICV) fractions')
    rpve = traits.Dict(traits.Float, desc='partial volume fractions')
    size = traits.Dict(traits.Int, desc='image sizes')
    spacing = traits.Tuple(traits.Float, traits.Float, traits.Float,
                           desc='image sizes')
    snr = traits.Dict(traits.Float)
    cnr = traits.Float
    fber = traits.Float
    efc = traits.Float
    art_qi1 = traits.Float
    out_qc = traits.Dict(desc='output flattened dictionary with all measures')


class StructuralQC(BaseInterface):
    """
    Computes anatomical :abr:`QC (Quality Control)` measures on the
    structural image given as input

    """
    input_spec = StructuralQCInputSpec
    output_spec = StructuralQCOutputSpec

    def __init__(self, **inputs):
        self._results = self.output_spec().get()
        super(StructuralQC, self).__init__(**inputs)

    def _run_interface(self, runtime):
        imnii = nb.load(self.inputs.in_file)
        imdata = np.nan_to_num(imnii.get_data())

        # Cast to float32
        if np.issubdtype(imdata.dtype, (np.float32, np.int32, np.uint8)):
            imdata = imdata.astype(np.float32)
        else:
            raise RuntimeError('Error: unsupported datatype %s' % imdata.dtype)

        # Remove negative values
        imdata[imdata < 0] = 0

        segnii = nb.load(self.inputs.in_segm)
        segdata = segnii.get_data().astype(np.uint8)

        # SNR
        snrvals = []
        for tlabel in ['csf', 'wm', 'gm']:
            snrvals.append(snr(imdata, segdata, tlabel))
            self._results['snr'][tlabel] = snrvals[-1]
        self._results['snr']['total'] = np.mean(snrvals)

        # CNR
        self._results['cnr'] = cnr(imdata, segdata)

        # FBER
        self._results['fber'] = fber(imdata, segdata)

        # EFC
        self._results['efc'] = efc(imdata)

        # Artifacts
        self._results['qi1'] = artifacts(imdata, segdata)[0]

        pvmdata = []
        for fname in self.inputs.in_pvms:
            pvmdata.append(nb.load(fname).get_data().astype(np.float32))

        # ICVs
        self._results['icvs'] = volume_fraction(pvmdata)

        # RPVE
        self._results['rpve'] = rpve(pvmdata, segdata)

        # Summary stats
        mean, stdv, p95, p05 = summary_stats(imdata, pvmdata)
        self._results.update(
            {'mean': mean, 'stdv': stdv, 'p95': p95, 'p05': p05})

        # Image specs
        self._results['size'] = {'x': imdata.shape[0],
                                 'y': imdata.shape[1],
                                 'z': imdata.shape[2]}
        self._results['spacing'] = {
            i: v for i, v in zip(['x', 'y', 'z'],
                                 imdata.get_header().get_zooms()[:3])}

        try:
            self._results['size']['t'] = imdata.shape[3]
        except IndexError:
            pass

        try:
            self._results['spacing']['tr'] = imdata.get_header().get_zooms()[3]
        except IndexError:
            pass

        # Bias
        bias = nb.load(self.inputs.in_bias).get_data()[segdata > 0]
        self._results['bias'] = {
            'max': bias.max(), 'min': bias.min(), 'med': np.median(bias)}  #pylint: disable=E1101


        # Flatten the dictionary
        out_qc = {}
        for k, value in list(self._results.items()):
            if not isinstance(value, dict):
                out_qc[k] = value
            else:
                for subk, subval in list(value.items()):
                    out_qc['%s_%s' % (k, subk)] = subval
        self._results['out_qc'] = out_qc

        return runtime

    def _list_outputs(self):
        return self._results
