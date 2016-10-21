#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# @Author: oesteban
# @Date:   2016-01-05 11:29:40
# @Email:  code@oscaresteban.es
# @Last modified by:   oesteban
# @Last Modified time: 2016-10-17 10:02:48
""" Visualization interfaces """
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from __future__ import unicode_literals

import os.path as op
from nipype.interfaces.base import (BaseInterface, traits, TraitedSpec, File,
                                    OutputMultiPath, BaseInterfaceInputSpec,
                                    isdefined)
from mriqc.utils.misc import split_ext
from mriqc.interfaces.viz_utils import (plot_mosaic, plot_fd, plot_segmentation)

class PlotContoursInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True,
                   desc='File to be plotted')
    in_contours = File(exists=True, mandatory=True,
                       desc='file to pick the contours from')
    cut_coords = traits.Int(8, usedefault=True, desc='number of slices')
    levels = traits.List([.5], traits.Float, usedefault=True,
                         desc='add a contour per level')
    colors = traits.List(['r'], traits.Str, usedefault=True,
                         desc='colors to be used for contours')
    display_mode = traits.Enum('ortho', 'x', 'y', 'z', 'yx', 'xz', 'yz', usedefault=True,
                               desc='visualization mode')
    saturate = traits.Bool(False, usedefault=True, desc='saturate background')
    out_file = traits.File(exists=False, desc='output file name')

class PlotContoursOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='output svg file')

class PlotContours(BaseInterface):
    """ Plot contours """
    input_spec = PlotContoursInputSpec
    output_spec = PlotContoursOutputSpec

    def __init__(self, **inputs):
        self._results = {}
        super(PlotContours, self).__init__(**inputs)

    def _list_outputs(self):
        return self._results

    def _run_interface(self, runtime):
        out_file = None
        if isdefined(self.inputs.out_file):
            out_file = self.inputs.out_file

        fname, _ = split_ext(self.inputs.in_file, out_file)
        out_file = op.abspath('plot_' + fname + '_contours.svg')
        self._results['out_file'] = out_file

        plot_segmentation(
            self.inputs.in_file,
            self.inputs.in_contours,
            out_file=out_file,
            cut_coords=self.inputs.cut_coords,
            display_mode=self.inputs.display_mode,
            levels=self.inputs.levels,
            colors=self.inputs.colors,
            saturate=self.inputs.saturate)

        return runtime

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
            title += ', subject {}'.format(self.inputs.subject)

        if isdefined(self.inputs.metadata):
            title += ' (' + '_'.join(self.inputs.metadata) + ')'

        figsize = None
        if isdefined(self.inputs.figsize):
            figsize = self.inputs.figsize

        fig = plot_mosaic(
            self.inputs.in_file,
            title=title,
            overlay_mask=mask,
            figsize=figsize)

        fig.savefig(self.inputs.out_file, dpi=self.inputs.dpi)
        fig.clf()
        return runtime

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['out_file'] = op.abspath(self.inputs.out_file)
        return outputs


class PlotFDInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True,
                   desc='File to be plotted')
    fd_radius = traits.Float(80., mandatory=True, usedefault=True,
                             desc='Radius to compute power of FD')
    figsize = traits.Tuple(
        (8.27, 3.0), traits.Float, traits.Float, usedefault=True,
        desc='Figure size')
    dpi = traits.Int(300, usedefault=True, desc='Desired DPI of figure')
    out_file = File('fd_power_2012.pdf', usedefault=True, desc='output file name')


class PlotFDOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='output pdf file')


class PlotFD(BaseInterface):
    """
    Plots the frame displacement of a dataset
    """

    input_spec = PlotFDInputSpec
    output_spec = PlotFDOutputSpec

    def _run_interface(self, runtime):

        if isdefined(self.inputs.figsize):
            fig = plot_fd(
                self.inputs.in_file,
                self.inputs.fd_radius,
                figsize=self.inputs.figsize)
        else:
            fig = plot_fd(self.inputs.in_file,
                          self.inputs.fd_radius)

        fig.savefig(self.inputs.out_file, dpi=float(self.inputs.dpi))

        return runtime

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['out_file'] = op.abspath(self.inputs.out_file)
        return outputs


# class ReportInputSpec(BaseInterfaceInputSpec):
#     in_csv = File(exists=True, mandatory=True, desc='File to be plotted')
#     qctype = traits.Enum('anatomical', 'functional', mandatory=True, desc='Type of report')
#     sub_list = traits.List([], traits.Tuple(traits.Str(), traits.Str(), traits.Str(), traits.Str()),
#                            usedefault=True, desc='List of subjects requested')
#     settings = traits.Dict(desc='Settings')


# class ReportOutputSpec(TraitedSpec):
#     out_group = File(exists=True, desc='output pdf file, group report')
#     out_indiv = OutputMultiPath(File(exists=True), desc='individual reports')


# class Report(BaseInterface):
#     input_spec = ReportInputSpec
#     output_spec = ReportOutputSpec

#     def _run_interface(self, runtime):
#         from mriqc.reports import workflow_report
#         settings = None
#         if isdefined(self.inputs.settings):
#             settings = self.inputs.settings
#         self._results = workflow_report(self.inputs.in_csv, self.inputs.qctype,
#                                         self.inputs.sub_list, settings)
#         return runtime

#     def _list_outputs(self):
#         outputs = self.output_spec().get()
#         outputs['out_group'] = self._results[0]
#         outputs['out_indiv'] = self._results[1]
#         return outputs


