#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

from __future__ import print_function, division, absolute_import, unicode_literals
from os import path as op
import numpy as np
import nibabel as nb

from .base import MRIQCBaseInterface
from nipype.interfaces.base import traits, TraitedSpec, BaseInterfaceInputSpec, File, isdefined
from nilearn.signal import clean


class SpikesInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True, desc='input fMRI dataset')
    in_mask = File(exists=True, desc='brain mask')
    spike_thresh = traits.Float(6., usedefault=True,
                                desc='z-score to call one timepoint of one axial slice a spike')
    skip_frames = traits.Int(4, usedefault=True,
                             desc='number of frames to skip in the beginning of the time series')


class SpikesOutputSpec(TraitedSpec):
    out_brain_tsz = File(desc='slice-wise z-scored timeseries (Z x N), inside brainmask')
    out_spikes = File(desc='indices of spikes')
    out_bg_tsz = File(desc='slice-wise z-scored timeseries (Z x N), only background')
    num_spikes = traits.Int(desc='number of spikes found (total)')

class Spikes(MRIQCBaseInterface):

    """
    Computes the number of spikes
    https://github.com/cni/nims/blob/master/nimsproc/qa_report.py

    """
    input_spec = SpikesInputSpec
    output_spec = SpikesOutputSpec

    def _run_interface(self, runtime):
        func_nii = nb.load(self.inputs.in_file)
        func_data = func_nii.get_data()
        func_shape = func_data.shape
        ntsteps = func_shape[-1]
        tr = func_nii.get_header().get_zooms()[-1]
        nskip = self.inputs.skip_frames

        background = None
        if isdefined(self.inputs.in_mask):
            mask_data = nb.load(self.inputs.in_mask).get_data()
            mask_data[...,:nskip] = 0

            data = func_data.reshape(-1, ntsteps)
            clean_data = clean(data[:, nskip:].T, t_r=tr, standardize=False).T
            new_shape = (*func_shape[:-1], clean_data.shape[-1])
            func_data = np.zeros(func_shape)
            func_data[..., nskip:] = clean_data.reshape(new_shape)
            brain = np.ma.array(func_data,
                                mask=np.stack([mask_data!=1] * ntsteps, axis=-1))

            mask_data[...,:self.inputs.skip_frames] = 1
            background = np.ma.array(func_data, mask=np.stack([mask_data==1] * ntsteps,
                                     axis=-1))
        else:
            brain, mask_data, _ = auto_mask(func_data, nskip=self.inputs.skip_frames)

        global_ts = brain.mean(0).mean(0).mean(0)
        total_spikes, ts_z = find_spikes(brain - global_ts, self.inputs.spike_thresh)

        out_brain_tsz = op.abspath('brain_tsz.txt')
        self._results['out_brain_tsz'] = out_brain_tsz
        np.savetxt(out_brain_tsz, ts_z)

        if not background is None:
            global_bg = background.mean(0).mean(0).mean(0)
            bg_spikes, ts_z_bg = find_spikes(background - global_bg, self.inputs.spike_thresh)

            out_bg_tsz = op.abspath('bg_tsz.txt')
            self._results['out_bg_tsz'] = out_bg_tsz
            np.savetxt(out_bg_tsz, ts_z_bg)
            total_spikes += bg_spikes

        total_spikes = list(set(total_spikes))
        out_spikes = op.abspath('spike_index.txt')
        self._results['out_spikes'] = out_spikes
        np.savetxt(out_spikes, total_spikes)
        self._results['num_spikes'] = len(total_spikes)

        return runtime


def find_spikes(data, spike_thresh):
    slice_mean = data.mean(axis=0).mean(axis=0)
    t_z = (slice_mean - np.atleast_2d(slice_mean.mean(axis=1)).T) / np.atleast_2d(
        slice_mean.std(axis=1)).T
    spikes = np.abs(t_z) > spike_thresh
    spike_inds = np.transpose(spikes.nonzero())
    # mask out the spikes and recompute z-scores using variance uncontaminated with spikes.
    # This will catch smaller spikes that may have been swamped by big ones.
    data.mask[:, :, spike_inds[:, 0], spike_inds[:, 1]] = True
    slice_mean2 = data.mean(axis=0).mean(axis=0)
    t_z = (slice_mean - np.atleast_2d(slice_mean.mean(axis=1)).T) / np.atleast_2d(
        slice_mean2.std(axis=1)).T
    spikes = np.logical_or(spikes, np.abs(t_z) > spike_thresh)
    spike_inds = [tuple(i) for i in np.transpose(spikes.nonzero())]
    return spike_inds, t_z


def auto_mask(data, raw_d=None, nskip=3, mask_bad_end_vols=False):
    from dipy.segment.mask import median_otsu
    mn = data[:, :, :, nskip:].mean(3)
    masked_data, mask = median_otsu(mn, 3, 2)
    mask = np.concatenate((
        np.tile(True, (data.shape[0], data.shape[1], data.shape[2], nskip)),
        np.tile(np.expand_dims(mask == 0, 3), (1, 1, 1, data.shape[3]-nskip))),
        axis=3)
    mask_vols = np.zeros((mask.shape[-1]), dtype=int)
    if mask_bad_end_vols:
        # Some runs have corrupt volumes at the end (e.g., mux scans that are stopped prematurely). Mask those too.
        # But... motion correction might have interpolated the empty slices such that they aren't exactly zero.
        # So use the raw data to find these bad volumes.
        # 2015.10.29 RFD: this caused problems with some non-mux EPI scans that (inexplicably)
        # have empty slices at the top of the brain. So we'll disable it for
        # now.
        if raw_d is None:
            slice_max = data.max(0).max(0)
        else:
            slice_max = raw_d.max(0).max(0)

        bad = np.any(slice_max == 0, axis=0)
        # We don't want to miss a bad volume somewhere in the middle, as that could be a valid artifact.
        # So, only mask bad vols that are contiguous to the end.
        mask_vols = np.array([np.all(bad[i:]) for i in range(bad.shape[0])])
    # Mask out the skip volumes at the beginning
    mask_vols[0:nskip] = True
    mask[..., mask_vols] = True
    brain = np.ma.masked_array(data, mask=mask)
    good_vols = np.logical_not(mask_vols)
    return brain, mask, good_vols
