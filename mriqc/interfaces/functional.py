#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

from __future__ import print_function, division, absolute_import, unicode_literals
from os import path as op
import numpy as np
import nibabel as nb
from nilearn.signal import clean
from builtins import zip

from niworkflows.nipype.interfaces.base import (
    traits, TraitedSpec, File, isdefined, BaseInterfaceInputSpec,
    SimpleInterface
)
from niworkflows.nipype import logging

from ..utils.misc import _flatten_dict
from ..qc.anatomical import snr, fber, efc, summary_stats
from ..qc.functional import gsr
IFLOGGER = logging.getLogger('interface')


class FunctionalQCInputSpec(BaseInterfaceInputSpec):
    in_epi = File(exists=True, mandatory=True, desc='input EPI file')
    in_hmc = File(exists=True, mandatory=True, desc='input motion corrected file')
    in_tsnr = File(exists=True, mandatory=True, desc='input tSNR volume')
    in_mask = File(exists=True, mandatory=True, desc='input mask')
    direction = traits.Enum('all', 'x', 'y', '-x', '-y', usedefault=True,
                            desc='direction for GSR computation')
    in_fd = File(exists=True, mandatory=True, desc='motion parameters for FD computation')
    fd_thres = traits.Float(0.2, usedefault=True, desc='motion threshold for FD computation')
    in_dvars = File(exists=True, mandatory=True, desc='input file containing DVARS')
    in_fwhm = traits.List(traits.Float, mandatory=True,
                          desc='smoothness estimated with AFNI')


class FunctionalQCOutputSpec(TraitedSpec):
    fber = traits.Float
    efc = traits.Float
    snr = traits.Float
    gsr = traits.Dict
    tsnr = traits.Float
    dvars = traits.Dict
    fd = traits.Dict
    fwhm = traits.Dict(desc='full width half-maximum measure')
    size = traits.Dict
    spacing = traits.Dict
    summary = traits.Dict

    out_qc = traits.Dict(desc='output flattened dictionary with all measures')


class FunctionalQC(SimpleInterface):
    """
    Computes anatomical :abbr:`QC (Quality Control)` measures on the
    structural image given as input

    """
    input_spec = FunctionalQCInputSpec
    output_spec = FunctionalQCOutputSpec

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
        mskdata[mskdata < 0] = 0
        mskdata[mskdata > 0] = 1
        mskdata = mskdata.astype(np.uint8)

        # Summary stats
        stats = summary_stats(epidata, mskdata, erode=True)
        self._results['summary'] = stats

        # SNR
        self._results['snr'] = snr(stats['fg']['median'], stats['fg']['stdv'], stats['fg']['n'])
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

        # DVARS
        dvars_avg = np.loadtxt(self.inputs.in_dvars, skiprows=1,
                               usecols=list(range(3))).mean(axis=0)
        dvars_col = ['std', 'nstd', 'vstd']
        self._results['dvars'] = {
            key: float(val) for key, val in zip(dvars_col, dvars_avg)
        }

        # tSNR
        tsnr_data = nb.load(self.inputs.in_tsnr).get_data()
        self._results['tsnr'] = float(np.median(tsnr_data[mskdata > 0]))

        # FD
        fd_data = np.loadtxt(self.inputs.in_fd, skiprows=1)
        num_fd = np.float((fd_data > self.inputs.fd_thres).sum())
        self._results['fd'] = {
            'mean': float(fd_data.mean()),
            'num': int(num_fd),
            'perc': float(num_fd * 100 / (len(fd_data) + 1))
        }

        # FWHM
        fwhm = np.array(self.inputs.in_fwhm[:3]) / np.array(hmcnii.get_header().get_zooms()[:3])
        self._results['fwhm'] = {
            'x': float(fwhm[0]), 'y': float(fwhm[1]), 'z': float(fwhm[2]),
            'avg': float(np.average(fwhm))}

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


class SpikesInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True, desc='input fMRI dataset')
    in_mask = File(exists=True, desc='brain mask')
    invert_mask = traits.Bool(False, usedefault=True, desc='invert mask')
    no_zscore = traits.Bool(False, usedefault=True, desc='do not zscore')
    detrend = traits.Bool(True, usedefault=True, desc='do detrend')
    spike_thresh = traits.Float(6., usedefault=True,
                                desc='z-score to call one timepoint of one axial slice a spike')
    skip_frames = traits.Int(0, usedefault=True,
                             desc='number of frames to skip in the beginning of the time series')
    out_tsz = File('spikes_tsz.txt', usedefault=True, desc='output file name')
    out_spikes = File(
        'spikes_idx.txt', usedefault=True, desc='output file name')


class SpikesOutputSpec(TraitedSpec):
    out_tsz = File(
        desc='slice-wise z-scored timeseries (Z x N), inside brainmask')
    out_spikes = File(desc='indices of spikes')
    num_spikes = traits.Int(desc='number of spikes found (total)')


class Spikes(SimpleInterface):

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

        if self.inputs.detrend:
            data = func_data.reshape(-1, ntsteps)
            clean_data = clean(data[:, nskip:].T, t_r=tr, standardize=False).T
            new_shape = (
                func_shape[0], func_shape[1], func_shape[2], clean_data.shape[-1])
            func_data = np.zeros(func_shape)
            func_data[..., nskip:] = clean_data.reshape(new_shape)

        if not isdefined(self.inputs.in_mask):
            _, mask_data, _ = auto_mask(
                func_data, nskip=self.inputs.skip_frames)
        else:
            mask_data = nb.load(self.inputs.in_mask).get_data()
            mask_data[..., :nskip] = 0
            mask_data = np.stack([mask_data] * ntsteps, axis=-1)

        if not self.inputs.invert_mask:
            brain = np.ma.array(func_data, mask=(mask_data != 1))
        else:
            mask_data[..., :self.inputs.skip_frames] = 1
            brain = np.ma.array(func_data, mask=(mask_data == 1))

        if self.inputs.no_zscore:
            ts_z = find_peaks(brain)
            total_spikes = []
        else:
            total_spikes, ts_z = find_spikes(
                brain, self.inputs.spike_thresh)
        total_spikes = list(set(total_spikes))

        out_tsz = op.abspath(self.inputs.out_tsz)
        self._results['out_tsz'] = out_tsz
        np.savetxt(out_tsz, ts_z)

        out_spikes = op.abspath(self.inputs.out_spikes)
        self._results['out_spikes'] = out_spikes
        np.savetxt(out_spikes, total_spikes)
        self._results['num_spikes'] = len(total_spikes)
        return runtime


def find_peaks(data):
    t_z = [data[:, :, i, :].mean(axis=0).mean(axis=0) for i in range(data.shape[2])]
    return t_z


def find_spikes(data, spike_thresh):
    data -= np.median(np.median(np.median(data, axis=0), axis=0), axis=0)
    slice_mean = np.median(np.median(data, axis=0), axis=0)
    t_z = _robust_zscore(slice_mean)
    spikes = np.abs(t_z) > spike_thresh
    spike_inds = np.transpose(spikes.nonzero())
    # mask out the spikes and recompute z-scores using variance uncontaminated with spikes.
    # This will catch smaller spikes that may have been swamped by big
    # ones.
    data.mask[:, :, spike_inds[:, 0], spike_inds[:, 1]] = True
    slice_mean2 = np.median(np.median(data, axis=0), axis=0)
    t_z = _robust_zscore(slice_mean2)

    spikes = np.logical_or(spikes, np.abs(t_z) > spike_thresh)
    spike_inds = [tuple(i) for i in np.transpose(spikes.nonzero())]
    return spike_inds, t_z


def auto_mask(data, raw_d=None, nskip=3, mask_bad_end_vols=False):
    from dipy.segment.mask import median_otsu
    mn = data[:, :, :, nskip:].mean(3)
    _, mask = median_otsu(mn, 3, 2)  # oesteban: masked_data was not used
    mask = np.concatenate((
        np.tile(True, (data.shape[0], data.shape[1], data.shape[2], nskip)),
        np.tile(np.expand_dims(mask == 0, 3), (1, 1, 1, data.shape[3] - nskip))),
        axis=3)
    mask_vols = np.zeros((mask.shape[-1]), dtype=int)
    if mask_bad_end_vols:
        # Some runs have corrupt volumes at the end (e.g., mux scans that are stopped
        # prematurely). Mask those too.
        # But... motion correction might have interpolated the empty slices such that
        # they aren't exactly zero.
        # So use the raw data to find these bad volumes.
        # 2015.10.29 RFD: this caused problems with some non-mux EPI scans that (inexplicably)
        # have empty slices at the top of the brain. So we'll disable it for
        # now.
        if raw_d is None:
            slice_max = data.max(0).max(0)
        else:
            slice_max = raw_d.max(0).max(0)

        bad = np.any(slice_max == 0, axis=0)
        # We don't want to miss a bad volume somewhere in the middle,
        # as that could be a valid artifact.
        # So, only mask bad vols that are contiguous to the end.
        mask_vols = np.array([np.all(bad[i:]) for i in range(bad.shape[0])])
    # Mask out the skip volumes at the beginning
    mask_vols[0:nskip] = True
    mask[..., mask_vols] = True
    brain = np.ma.masked_array(data, mask=mask)
    good_vols = np.logical_not(mask_vols)
    return brain, mask, good_vols


def _robust_zscore(data):
    return ((data - np.atleast_2d(np.median(data, axis=1)).T) /
            np.atleast_2d(data.std(axis=1)).T)
