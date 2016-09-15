#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# @Author: oesteban
# @Date:   2016-01-05 11:29:40
# @Email:  code@oscaresteban.es
# @Last modified by:   oesteban
# @Last Modified time: 2016-04-13 08:10:35
""" Nipype interfaces to support anatomical workflow """
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from __future__ import unicode_literals
from mriqc.qc.functional import compute_dvars

from nipype.interfaces.base import traits, TraitedSpec, BaseInterface, BaseInterfaceInputSpec, File


class ComputeDVARSInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True, desc='functional data, after HMC')
    in_mask = File(exists=True, mandatory=True, desc='a brain mask')
    output_all = traits.Bool(False, usedefault=True, desc='output all DVARS')


class ComputeDVARSOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='output text file')


class ComputeDVARS(BaseInterface):
    """
    Computes the DVARS.
    """
    input_spec = ComputeDVARSInputSpec
    output_spec = ComputeDVARSOutputSpec

    def __init__(self, **inputs):
        self._results = {}
        super(ComputeDVARS, self).__init__(**inputs)

    def _run_interface(self, runtime):
        self._results['out_file'] = compute_dvars(self.inputs.in_file,
                                                  self.inputs.in_mask,
                                                  output_all=self.inputs.output_all)
        return runtime

    def _list_outputs(self):
        return self._results


