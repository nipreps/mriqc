#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# @Author: oesteban
# @Date:   2016-01-05 11:29:40
# @Email:  code@oscaresteban.es
# @Last modified by:   oesteban
""" Visualization interfaces """

from pathlib import Path
import numpy as np
from nipype.interfaces.base import (
    traits, TraitedSpec, File, BaseInterfaceInputSpec, isdefined,
    SimpleInterface)

from io import open  # pylint: disable=W0622
from ..viz.utils import (plot_mosaic, plot_segmentation, plot_spikes)


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
    vmin = traits.Float(desc='minimum intensity')
    vmax = traits.Float(desc='maximum intensity')


class PlotContoursOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='output svg file')


class PlotContours(SimpleInterface):
    """ Plot contours """
    input_spec = PlotContoursInputSpec
    output_spec = PlotContoursOutputSpec

    def _run_interface(self, runtime):
        in_file_ref = Path(self.inputs.in_file)

        if isdefined(self.inputs.out_file):
            in_file_ref = Path(self.inputs.out_file)

        fname = in_file_ref.name.rstrip(
            ''.join(in_file_ref.suffixes))
        out_file = (Path(runtime.cwd) / ('plot_%s_contours.svg' % fname)).resolve()
        self._results['out_file'] = str(out_file)

        vmax = None if not isdefined(self.inputs.vmax) else self.inputs.vmax
        vmin = None if not isdefined(self.inputs.vmin) else self.inputs.vmin

        plot_segmentation(
            self.inputs.in_file,
            self.inputs.in_contours,
            out_file=str(out_file),
            cut_coords=self.inputs.cut_coords,
            display_mode=self.inputs.display_mode,
            levels=self.inputs.levels,
            colors=self.inputs.colors,
            saturate=self.inputs.saturate,
            vmin=vmin, vmax=vmax)

        return runtime


class PlotBaseInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True,
                   desc='File to be plotted')
    title = traits.Str(desc='a title string for the plot')
    annotate = traits.Bool(True, usedefault=True, desc='annotate left/right')
    figsize = traits.Tuple(
        (11.69, 8.27), traits.Float, traits.Float, usedefault=True,
        desc='Figure size')
    dpi = traits.Int(300, usedefault=True, desc='Desired DPI of figure')
    out_file = File('mosaic.svg', usedefault=True, desc='output file name')
    cmap = traits.Str('Greys_r', usedefault=True)


class PlotMosaicInputSpec(PlotBaseInputSpec):
    bbox_mask_file = File(exists=True, desc='brain mask')
    only_noise = traits.Bool(False, desc='plot only noise')


class PlotMosaicOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='output pdf file')


class PlotMosaic(SimpleInterface):

    """
    Plots slices of a 3D volume into a pdf file
    """
    input_spec = PlotMosaicInputSpec
    output_spec = PlotMosaicOutputSpec

    def _run_interface(self, runtime):
        mask = None
        if isdefined(self.inputs.bbox_mask_file):
            mask = self.inputs.bbox_mask_file

        title = None
        if isdefined(self.inputs.title):
            title = self.inputs.title

        plot_mosaic(
            self.inputs.in_file,
            out_file=self.inputs.out_file,
            title=title,
            only_plot_noise=self.inputs.only_noise,
            bbox_mask_file=mask,
            cmap=self.inputs.cmap,
            annotate=self.inputs.annotate)
        self._results['out_file'] = str((Path(runtime.cwd) / self.inputs.out_file).resolve())
        return runtime


class PlotSpikesInputSpec(PlotBaseInputSpec):
    in_spikes = File(exists=True, mandatory=True, desc='tsv file of spikes')
    in_fft = File(exists=True, mandatory=True, desc='nifti file with the 4D FFT')


class PlotSpikesOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='output svg file')


class PlotSpikes(SimpleInterface):
    """
    Plot slices of a dataset with spikes
    """
    input_spec = PlotSpikesInputSpec
    output_spec = PlotSpikesOutputSpec

    def _run_interface(self, runtime):
        out_file = str((Path(runtime.cwd) / self.inputs.out_file).resolve())
        self._results['out_file'] = out_file

        spikes_list = np.loadtxt(self.inputs.in_spikes, dtype=int).tolist()
        # No spikes
        if not spikes_list:
            with open(out_file, 'w') as f:
                f.write('<p>No high-frequency spikes were found in this dataset</p>')
            return runtime

        spikes_list = [tuple(i) for i in np.atleast_2d(spikes_list).tolist()]
        plot_spikes(
            self.inputs.in_file, self.inputs.in_fft, spikes_list,
            out_file=out_file)
        return runtime
