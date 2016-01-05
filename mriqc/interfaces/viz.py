#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# @Author: oesteban
# @Date:   2016-01-05 11:29:40
# @Email:  code@oscaresteban.es
# @Last modified by:   oesteban
# @Last Modified time: 2016-01-05 11:32:27

import os
import os.path as op

import nibabel as nb
import numpy as np

from nipype.interfaces.base import (BaseInterface, traits, TraitedSpec, File,
                                    InputMultiPath, OutputMultiPath,
                                    BaseInterfaceInputSpec, isdefined,
                                    DynamicTraitedSpec, Undefined)

from .viz_utils import (plot_mosaic, plot_fd)

from nipype import logging
iflogger = logging.getLogger('interface')


class PlotMosaicInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True,
                   desc='File to be plotted')
    in_mask = File(exists=True, desc='Overlay mask')
    title = traits.Str('Volume', usedefault=True,
                       desc='modality name to be prepended')
    subject = traits.Str(desc='Subject id')
    metadata = traits.List(traits.Str, desc='additional metadata')
    figsize = traits.Tuple(
        (11.69, 8.27), traits.Float, traits.Float, usedefault=True,
        desc='Figure size')
    dpi = traits.Int(300, usedefault=True, desc='Desired DPI of figure')
    out_file = File('mosaic.pdf', usedefault=True, desc='output file name')


class PlotMosaicOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='output pdf file')


class PlotMosaic(BaseInterface):

    """
    Plots slices of a 3D volume into a pdf file
    """
    input_spec = PlotMosaicInputSpec
    output_spec = PlotMosaicOutputSpec

    def _run_interface(self, runtime):
        mask = None
        if isdefined(self.inputs.in_mask):
            mask = self.inputs.in_mask

        title = self.inputs.title
        if isdefined(self.inputs.subject):
            title += ', subject %s' % self.inputs.subject

        if isdefined(self.inputs.metadata):
            title += ' (' + '_'.join(self.inputs.metadata) + ')'

        if isdefined(self.inputs.figsize):
            fig = plot_mosaic(
                self.inputs.in_file,
                title=title,
                overlay_mask=mask,
                figsize=self.inputs.figsize)
        else:
            fig = plot_mosaic(
                self.inputs.in_file,
                title=title,
                overlay_mask=mask)

        fig.savefig(self.inputs.out_file, dpi=self.inputs.dpi)

        return runtime

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['out_file'] = op.abspath(self.inputs.out_file)
        return outputs


class PlotFDInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True,
                   desc='File to be plotted')
    title = traits.Str('FD', usedefault=True,
                       desc='modality name to be prepended')
    subject = traits.Str(desc='Subject id')
    metadata = traits.List(traits.Str, desc='additional metadata')
    figsize = traits.Tuple(
        (8.27, 3.0), traits.Float, traits.Float, usedefault=True,
        desc='Figure size')
    dpi = traits.Int(300, usedefault=True, desc='Desired DPI of figure')
    out_file = File('fd.pdf', usedefault=True, desc='output file name')


class PlotFDOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='output pdf file')


class PlotFD(BaseInterface):

    """
    Plots the frame displacement of a dataset
    """
    input_spec = PlotFDInputSpec
    output_spec = PlotFDOutputSpec

    def _run_interface(self, runtime):
        title = self.inputs.title
        if isdefined(self.inputs.subject):
            title += ', subject %s' % self.inputs.subject

        if isdefined(self.inputs.metadata):
            title += ' (' + '_'.join(self.inputs.metadata) + ')'

        if isdefined(self.inputs.figsize):
            fig = plot_fd(
                self.inputs.in_file,
                title=title,
                figsize=self.inputs.figsize)
        else:
            fig = plot_fd(self.inputs.in_file)

        fig.savefig(self.inputs.out_file, dpi=float(self.inputs.dpi))

        return runtime

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['out_file'] = op.abspath(self.inputs.out_file)
        return outputs
