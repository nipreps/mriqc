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
from __future__ import print_function, division, absolute_import, unicode_literals

import os.path as op
import nibabel as nb
import numpy as np
from nipype.interfaces.base import (traits, TraitedSpec, File,
                                    BaseInterfaceInputSpec, isdefined)
from io import open # pylint: disable=W0622
from mriqc.utils.misc import split_ext
from mriqc.viz.utils import (plot_mosaic_helper, plot_segmentation)
from mriqc.interfaces.base import MRIQCBaseInterface


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

class PlotContours(MRIQCBaseInterface):
    """ Plot contours """
    input_spec = PlotContoursInputSpec
    output_spec = PlotContoursOutputSpec

    def _run_interface(self, runtime):
        out_file = None
        if isdefined(self.inputs.out_file):
            out_file = self.inputs.out_file

        fname, _ = split_ext(self.inputs.in_file, out_file)
        out_file = op.abspath('plot_' + fname + '_contours.svg')
        self._results['out_file'] = out_file

        vmax = None if not isdefined(self.inputs.vmax) else self.inputs.vmax
        vmin = None if not isdefined(self.inputs.vmin) else self.inputs.vmin

        plot_segmentation(
            self.inputs.in_file,
            self.inputs.in_contours,
            out_file=out_file,
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


class PlotMosaic(MRIQCBaseInterface):

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

        plot_mosaic_helper(
            self.inputs.in_file,
            out_file=self.inputs.out_file,
            title=title,
            only_plot_noise=self.inputs.only_noise,
            bbox_mask_file=mask,
            cmap=self.inputs.cmap)
        self._results['out_file'] = op.abspath(self.inputs.out_file)
        return runtime


class PlotSpikesInputSpec(PlotBaseInputSpec):
    in_spikes = File(exists=True, mandatory=True, desc='tsv file of spikes')


class PlotSpikesOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='output svg file')


class PlotSpikes(MRIQCBaseInterface):
    """
    Plot slices of a dataset with spikes
    """
    input_spec = PlotSpikesInputSpec
    output_spec = PlotSpikesOutputSpec

    def _run_interface(self, runtime):
        out_file = op.abspath(self.inputs.out_file)
        self._results['out_file'] = out_file

        spikes_list = np.loadtxt(self.inputs.in_spikes, dtype=int)
        # No spikes
        if len(spikes_list) == 0:
            with open(out_file, 'w') as f:
                f.write('<p>No high-frequency spikes were found in this dataset</p>')
            return runtime

        spikes_list = [tuple(i) for i in np.atleast_2d(spikes_list).reshape(-1, 2)]

        # Spikes found
        nii = nb.load(self.inputs.in_file)
        data = nii.get_data()

        slices = []
        labels = []
        labelfmt = 't={0:.3f}s (z={1:d})'.format
        for t, z in spikes_list:
            if t > 0:
                slices.append(data[..., z, t - 1])
                labels.append(labelfmt(t - 1, z))
            slices.append(data[..., z, t])
            labels.append(labelfmt(t, z))

            if t < (len(spikes_list) - 1):
                slices.append(data[..., z, t + 1])
                labels.append(labelfmt(t + 1, z))

        spikes_data = np.stack(slices, axis=-1)
        nb.Nifti1Image(spikes_data, nii.get_affine(),
                       nii.get_header()).to_filename('spikes.nii.gz')

        title = None
        if isdefined(self.inputs.title):
            title = self.inputs.title

        # tr = nii.get_header().get_zooms()[-1]
        plot_mosaic_helper(
            op.abspath('spikes.nii.gz'),
            out_file=out_file,
            title=title,
            cmap=self.inputs.cmap,
            plot_sagittal=False,
            only_plot_noise=False,
            labels=labels)
        return runtime
