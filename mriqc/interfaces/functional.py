# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# Copyright 2021 The NiPreps Developers <nipreps@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# We support and encourage derived works from this project, please read
# about our expectations at
#
#     https://www.nipreps.org/community/licensing/
#
from os import path as op

import nibabel as nb
import numpy as np
from mriqc.qc.anatomical import efc, fber, snr, summary_stats
from mriqc.qc.functional import gsr
from mriqc.utils.misc import _flatten_dict
from nipype.interfaces.base import (
    BaseInterfaceInputSpec,
    File,
    SimpleInterface,
    TraitedSpec,
    traits,
)


class FunctionalQCInputSpec(BaseInterfaceInputSpec):
    in_epi = File(exists=True, mandatory=True, desc="input EPI file")
    in_hmc = File(exists=True, mandatory=True, desc="input motion corrected file")
    in_tsnr = File(exists=True, mandatory=True, desc="input tSNR volume")
    in_mask = File(exists=True, mandatory=True, desc="input mask")
    direction = traits.Enum(
        "all",
        "x",
        "y",
        "-x",
        "-y",
        usedefault=True,
        desc="direction for GSR computation",
    )
    in_fd = File(
        exists=True,
        mandatory=True,
        desc="motion parameters for FD computation",
    )
    fd_thres = traits.Float(0.2, usedefault=True, desc="motion threshold for FD computation")
    in_dvars = File(exists=True, mandatory=True, desc="input file containing DVARS")
    in_fwhm = traits.List(traits.Float, mandatory=True, desc="smoothness estimated with AFNI")


class FunctionalQCOutputSpec(TraitedSpec):
    fber = traits.Float
    efc = traits.Float
    snr = traits.Float
    gsr = traits.Dict
    tsnr = traits.Float
    dvars = traits.Dict
    fd = traits.Dict
    fwhm = traits.Dict(desc="full width half-maximum measure")
    size = traits.Dict
    spacing = traits.Dict
    summary = traits.Dict

    out_qc = traits.Dict(desc="output flattened dictionary with all measures")


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
        epidata = np.nan_to_num(np.float32(epinii.dataobj))
        epidata[epidata < 0] = 0

        # Get EPI data (with mc done) and get it ready
        hmcnii = nb.load(self.inputs.in_hmc)
        hmcdata = np.nan_to_num(np.float32(hmcnii.dataobj))
        hmcdata[hmcdata < 0] = 0

        # Get brain mask data
        msknii = nb.load(self.inputs.in_mask)
        mskdata = np.asanyarray(msknii.dataobj) > 0
        if np.sum(mskdata) < 100:
            raise RuntimeError(
                "Detected less than 100 voxels belonging to the brain mask. "
                "MRIQC failed to process this dataset."
            )

        # Summary stats
        rois = {"fg": mskdata.astype(np.uint8), "bg": (~mskdata).astype(np.uint8)}
        stats = summary_stats(epidata, rois)
        self._results["summary"] = stats

        # SNR
        self._results["snr"] = snr(stats["fg"]["median"], stats["fg"]["stdv"], stats["fg"]["n"])
        # FBER
        self._results["fber"] = fber(epidata, mskdata.astype(np.uint8))
        # EFC
        self._results["efc"] = efc(epidata)
        # GSR
        self._results["gsr"] = {}
        if self.inputs.direction == "all":
            epidir = ["x", "y"]
        else:
            epidir = [self.inputs.direction]

        for axis in epidir:
            self._results["gsr"][axis] = gsr(epidata, mskdata.astype(np.uint8), direction=axis)

        # DVARS
        dvars_avg = np.loadtxt(self.inputs.in_dvars, skiprows=1, usecols=list(range(3))).mean(
            axis=0
        )
        dvars_col = ["std", "nstd", "vstd"]
        self._results["dvars"] = {key: float(val) for key, val in zip(dvars_col, dvars_avg)}

        # tSNR
        tsnr_data = nb.load(self.inputs.in_tsnr).get_fdata()
        self._results["tsnr"] = float(np.median(tsnr_data[mskdata]))

        # FD
        fd_data = np.loadtxt(self.inputs.in_fd, skiprows=1)
        num_fd = np.float((fd_data > self.inputs.fd_thres).sum())
        self._results["fd"] = {
            "mean": float(fd_data.mean()),
            "num": int(num_fd),
            "perc": float(num_fd * 100 / (len(fd_data) + 1)),
        }

        # FWHM
        fwhm = np.array(self.inputs.in_fwhm[:3]) / np.array(hmcnii.header.get_zooms()[:3])
        self._results["fwhm"] = {
            "x": float(fwhm[0]),
            "y": float(fwhm[1]),
            "z": float(fwhm[2]),
            "avg": float(np.average(fwhm)),
        }

        # Image specs
        self._results["size"] = {
            "x": int(hmcdata.shape[0]),
            "y": int(hmcdata.shape[1]),
            "z": int(hmcdata.shape[2]),
        }
        self._results["spacing"] = {
            i: float(v) for i, v in zip(["x", "y", "z"], hmcnii.header.get_zooms()[:3])
        }

        try:
            self._results["size"]["t"] = int(hmcdata.shape[3])
        except IndexError:
            pass

        try:
            self._results["spacing"]["tr"] = float(hmcnii.header.get_zooms()[3])
        except IndexError:
            pass

        self._results["out_qc"] = _flatten_dict(self._results)
        return runtime


class SpikesInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True, desc="input fMRI dataset")
    in_mask = File(exists=True, mandatory=True, desc="brain mask")
    invert_mask = traits.Bool(False, usedefault=True, desc="invert mask")
    no_zscore = traits.Bool(False, usedefault=True, desc="do not zscore")
    detrend = traits.Bool(True, usedefault=True, desc="do detrend")
    spike_thresh = traits.Float(
        6.0,
        usedefault=True,
        desc="z-score to call one timepoint of one axial slice a spike",
    )
    skip_frames = traits.Int(
        0,
        usedefault=True,
        desc="number of frames to skip in the beginning of the time series",
    )
    out_tsz = File("spikes_tsz.txt", usedefault=True, desc="output file name")
    out_spikes = File("spikes_idx.txt", usedefault=True, desc="output file name")


class SpikesOutputSpec(TraitedSpec):
    out_tsz = File(desc="slice-wise z-scored timeseries (Z x N), inside brainmask")
    out_spikes = File(desc="indices of spikes")
    num_spikes = traits.Int(desc="number of spikes found (total)")


class Spikes(SimpleInterface):

    """
    Computes the number of spikes
    https://github.com/cni/nims/blob/master/nimsproc/qa_report.py

    """

    input_spec = SpikesInputSpec
    output_spec = SpikesOutputSpec

    def _run_interface(self, runtime):
        func_nii = nb.load(self.inputs.in_file)
        func_data = func_nii.get_fdata()
        func_shape = func_data.shape
        ntsteps = func_shape[-1]
        tr = func_nii.header.get_zooms()[-1]
        nskip = self.inputs.skip_frames

        if self.inputs.detrend:
            from nilearn.signal import clean

            data = func_data.reshape(-1, ntsteps)
            clean_data = clean(data[:, nskip:].T, t_r=tr, standardize=False).T
            new_shape = (
                func_shape[0],
                func_shape[1],
                func_shape[2],
                clean_data.shape[-1],
            )
            func_data = np.zeros(func_shape)
            func_data[..., nskip:] = clean_data.reshape(new_shape)

        mask_data = np.bool_(nb.load(self.inputs.in_mask).dataobj)
        mask_data[..., :nskip] = 0
        mask_data = np.stack([mask_data] * ntsteps, axis=-1)

        if not self.inputs.invert_mask:
            brain = np.ma.array(func_data, mask=(mask_data != 1))
        else:
            mask_data[..., : self.inputs.skip_frames] = 1
            brain = np.ma.array(func_data, mask=(mask_data == 1))

        if self.inputs.no_zscore:
            ts_z = find_peaks(brain)
            total_spikes = []
        else:
            total_spikes, ts_z = find_spikes(brain, self.inputs.spike_thresh)
        total_spikes = list(set(total_spikes))

        out_tsz = op.abspath(self.inputs.out_tsz)
        self._results["out_tsz"] = out_tsz
        np.savetxt(out_tsz, ts_z)

        out_spikes = op.abspath(self.inputs.out_spikes)
        self._results["out_spikes"] = out_spikes
        np.savetxt(out_spikes, total_spikes)
        self._results["num_spikes"] = len(total_spikes)
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


def _robust_zscore(data):
    return (data - np.atleast_2d(np.median(data, axis=1)).T) / np.atleast_2d(data.std(axis=1)).T
