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
from __future__ import annotations

from os import path as op

import nibabel as nb
import numpy as np
import pandas as pd
from nipype.interfaces.base import (
    BaseInterfaceInputSpec,
    File,
    InputMultiObject,
    SimpleInterface,
    TraitedSpec,
    Undefined,
    isdefined,
    traits,
)
from nipype.utils.misc import normalize_mc_params

from mriqc.qc.anatomical import efc, fber, snr, summary_stats
from mriqc.qc.functional import gsr
from mriqc.utils.misc import _flatten_dict


class FunctionalQCInputSpec(BaseInterfaceInputSpec):
    in_epi = File(exists=True, mandatory=True, desc='input EPI file')
    in_hmc = File(exists=True, mandatory=True, desc='input motion corrected file')
    in_tsnr = File(exists=True, mandatory=True, desc='input tSNR volume')
    in_mask = File(exists=True, mandatory=True, desc='input mask')
    direction = traits.Enum(
        'all',
        'x',
        'y',
        '-x',
        '-y',
        usedefault=True,
        desc='direction for GSR computation',
    )
    in_fd = File(
        exists=True,
        mandatory=True,
        desc='motion parameters for FD computation',
    )
    fd_thres = traits.Float(0.2, usedefault=True, desc='motion threshold for FD computation')
    in_dvars = File(exists=True, mandatory=True, desc='input file containing DVARS')
    in_fwhm = traits.List(traits.Float, mandatory=True, desc='smoothness estimated with AFNI')


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
                'Detected less than 100 voxels belonging to the brain mask. '
                'MRIQC failed to process this dataset.'
            )

        # Summary stats
        rois = {'fg': mskdata.astype(np.uint8), 'bg': (~mskdata).astype(np.uint8)}
        stats = summary_stats(epidata, rois)
        self._results['summary'] = stats

        # SNR
        self._results['snr'] = snr(stats['fg']['median'], stats['fg']['stdv'], stats['fg']['n'])
        # FBER
        self._results['fber'] = fber(epidata, mskdata.astype(np.uint8))
        # EFC
        self._results['efc'] = efc(epidata)
        # GSR
        self._results['gsr'] = {}
        if self.inputs.direction == 'all':
            epidir = ['x', 'y']
        else:
            epidir = [self.inputs.direction]

        for axis in epidir:
            self._results['gsr'][axis] = gsr(epidata, mskdata.astype(np.uint8), direction=axis)

        # DVARS
        dvars_avg = np.loadtxt(self.inputs.in_dvars, skiprows=1, usecols=list(range(3))).mean(
            axis=0
        )
        dvars_col = ['std', 'nstd', 'vstd']
        self._results['dvars'] = {key: float(val) for key, val in zip(dvars_col, dvars_avg)}

        # tSNR
        tsnr_data = nb.load(self.inputs.in_tsnr).get_fdata()
        self._results['tsnr'] = float(np.median(tsnr_data[mskdata]))

        # FD
        fd_data = np.loadtxt(self.inputs.in_fd, skiprows=1)
        num_fd = (fd_data > self.inputs.fd_thres).sum()
        self._results['fd'] = {
            'mean': float(fd_data.mean()),
            'num': int(num_fd),
            'perc': float(num_fd * 100 / (len(fd_data) + 1)),
        }

        # FWHM
        fwhm = np.array(self.inputs.in_fwhm[:3]) / np.array(hmcnii.header.get_zooms()[:3])
        self._results['fwhm'] = {
            'x': float(fwhm[0]),
            'y': float(fwhm[1]),
            'z': float(fwhm[2]),
            'avg': float(np.average(fwhm)),
        }

        # Image specs
        self._results['size'] = {
            'x': int(hmcdata.shape[0]),
            'y': int(hmcdata.shape[1]),
            'z': int(hmcdata.shape[2]),
        }
        self._results['spacing'] = {
            i: float(v) for i, v in zip(['x', 'y', 'z'], hmcnii.header.get_zooms()[:3])
        }

        try:
            self._results['size']['t'] = int(hmcdata.shape[3])
        except IndexError:
            pass

        try:
            self._results['spacing']['tr'] = float(hmcnii.header.get_zooms()[3])
        except IndexError:
            pass

        self._results['out_qc'] = _flatten_dict(self._results)
        return runtime


class SpikesInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True, desc='input fMRI dataset')
    in_mask = File(exists=True, mandatory=True, desc='brain mask')
    invert_mask = traits.Bool(False, usedefault=True, desc='invert mask')
    no_zscore = traits.Bool(False, usedefault=True, desc='do not zscore')
    detrend = traits.Bool(True, usedefault=True, desc='do detrend')
    spike_thresh = traits.Float(
        6.0,
        usedefault=True,
        desc='z-score to call one timepoint of one axial slice a spike',
    )
    skip_frames = traits.Int(
        0,
        usedefault=True,
        desc='number of frames to skip in the beginning of the time series',
    )
    out_tsz = File('spikes_tsz.txt', usedefault=True, desc='output file name')
    out_spikes = File('spikes_idx.txt', usedefault=True, desc='output file name')


class SpikesOutputSpec(TraitedSpec):
    out_tsz = File(desc='slice-wise z-scored timeseries (Z x N), inside brainmask')
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
        func_data = func_nii.get_fdata(dtype='float32')
        func_shape = func_data.shape
        ntsteps = func_shape[-1]
        tr = func_nii.header.get_zooms()[-1]
        nskip = self.inputs.skip_frames

        mask_data = np.bool_(nb.load(self.inputs.in_mask).dataobj)
        mask_data[..., :nskip] = 0
        mask_data = np.stack([mask_data] * ntsteps, axis=-1)

        if not self.inputs.invert_mask:
            brain = np.ma.array(func_data, mask=(mask_data != 1))
        else:
            mask_data[..., : self.inputs.skip_frames] = 1
            brain = np.ma.array(func_data, mask=(mask_data == 1))

        if self.inputs.detrend:
            from nilearn.signal import clean

            brain = clean(brain[:, nskip:].T, t_r=tr, standardize=False).T

        if self.inputs.no_zscore:
            ts_z = find_peaks(brain)
            total_spikes = []
        else:
            total_spikes, ts_z = find_spikes(brain, self.inputs.spike_thresh)
        total_spikes = list(set(total_spikes))

        out_tsz = op.abspath(self.inputs.out_tsz)
        self._results['out_tsz'] = out_tsz
        np.savetxt(out_tsz, ts_z)

        out_spikes = op.abspath(self.inputs.out_spikes)
        self._results['out_spikes'] = out_spikes
        np.savetxt(out_spikes, total_spikes)
        self._results['num_spikes'] = len(total_spikes)
        return runtime


class _SelectEchoInputSpec(BaseInterfaceInputSpec):
    in_files = InputMultiObject(File(exists=True), mandatory=True, desc='input EPI file(s)')
    metadata = InputMultiObject(traits.Dict(), desc='sidecar JSON files corresponding to in_files')
    te_reference = traits.Float(0.030, usedefault=True, desc='reference SE-EPI echo time')


class _SelectEchoOutputSpec(TraitedSpec):
    out_file = File(desc='selected echo')
    echo_index = traits.Int(desc='index of the selected echo')
    is_multiecho = traits.Bool(desc='whether it is a multiecho dataset')


class SelectEcho(SimpleInterface):
    """
    Computes anatomical :abbr:`QC (Quality Control)` measures on the
    structural image given as input

    """

    input_spec = _SelectEchoInputSpec
    output_spec = _SelectEchoOutputSpec

    def _run_interface(self, runtime):
        (
            self._results['out_file'],
            self._results['echo_index'],
        ) = select_echo(
            self.inputs.in_files,
            te_echos=(
                _get_echotime(self.inputs.metadata) if isdefined(self.inputs.metadata) else None
            ),
            te_reference=self.inputs.te_reference,
        )
        self._results['is_multiecho'] = self._results['echo_index'] != -1
        return runtime


class GatherTimeseriesInputSpec(TraitedSpec):
    dvars = File(exists=True, mandatory=True, desc='file containing DVARS')
    fd = File(exists=True, mandatory=True, desc='input framewise displacement')
    mpars = File(exists=True, mandatory=True, desc='input motion parameters')
    mpars_source = traits.Enum(
        'FSL',
        'AFNI',
        'SPM',
        'FSFAST',
        'NIPY',
        desc='Source of movement parameters',
        mandatory=True,
    )
    outliers = File(
        exists=True,
        mandatory=True,
        desc="input file containing timeseries of AFNI's outlier count",
    )
    quality = File(exists=True, mandatory=True, desc="input file containing AFNI's Quality Index")


class GatherTimeseriesOutputSpec(TraitedSpec):
    timeseries_file = File(desc='output confounds file')
    timeseries_metadata = traits.Dict(desc='Metadata dictionary describing columns')


class GatherTimeseries(SimpleInterface):
    """
    Gather quality metrics that are timeseries into one TSV file

    """

    input_spec = GatherTimeseriesInputSpec
    output_spec = GatherTimeseriesOutputSpec

    def _run_interface(self, runtime):
        # motion parameters
        mpars = np.apply_along_axis(
            func1d=normalize_mc_params,
            axis=1,
            arr=np.loadtxt(self.inputs.mpars),  # mpars is N_t x 6
            source=self.inputs.mpars_source,
        )
        timeseries = pd.DataFrame(
            mpars, columns=['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z']
        )

        # DVARS
        dvars = pd.read_csv(
            self.inputs.dvars,
            sep=r'\s+',
            skiprows=1,  # column names have spaces
            header=None,
            names=['dvars_std', 'dvars_nstd', 'dvars_vstd'],
        )
        dvars.index = pd.RangeIndex(1, timeseries.index.max() + 1)

        # FD
        fd = pd.read_csv(self.inputs.fd, sep=r'\s+', header=0, names=['framewise_displacement'])
        fd.index = pd.RangeIndex(1, timeseries.index.max() + 1)

        # AQI
        aqi = pd.read_csv(self.inputs.quality, sep=r'\s+', header=None, names=['aqi'])

        # Outliers
        aor = pd.read_csv(self.inputs.outliers, sep=r'\s+', header=None, names=['aor'])

        timeseries = pd.concat((timeseries, dvars, fd, aqi, aor), axis=1)

        timeseries_file = op.join(runtime.cwd, 'timeseries.tsv')

        timeseries.to_csv(timeseries_file, sep='\t', index=False, na_rep='n/a')

        self._results['timeseries_file'] = timeseries_file
        self._results['timeseries_metadata'] = _build_timeseries_metadata()
        return runtime


def _build_timeseries_metadata():
    return {
        'trans_x': {
            'LongName': 'Translation Along X Axis',
            'Description': 'Estimated Motion Parameter',
            'Units': 'mm',
        },
        'trans_y': {
            'LongName': 'Translation Along Y Axis',
            'Description': 'Estimated Motion Parameter',
            'Units': 'mm',
        },
        'trans_z': {
            'LongName': 'Translation Along Z Axis',
            'Description': 'Estimated Motion Parameter',
            'Units': 'mm',
        },
        'rot_x': {
            'LongName': 'Rotation Around X Axis',
            'Description': 'Estimated Motion Parameter',
            'Units': 'rad',
        },
        'rot_y': {
            'LongName': 'Rotation Around X Axis',
            'Description': 'Estimated Motion Parameter',
            'Units': 'rad',
        },
        'rot_z': {
            'LongName': 'Rotation Around X Axis',
            'Description': 'Estimated Motion Parameter',
            'Units': 'rad',
        },
        'dvars_std': {
            'LongName': 'Derivative of RMS Variance over Voxels, Standardized',
            'Description': (
                'Indexes the rate of change of BOLD signal across'
                'the entire brain at each frame of data, normalized with the'
                'standard deviation of the temporal difference time series'
            ),
        },
        'dvars_nstd': {
            'LongName': ('Derivative of RMS Variance over Voxels, Non-Standardized'),
            'Description': (
                'Indexes the rate of change of BOLD signal across'
                'the entire brain at each frame of data, not normalized.'
            ),
        },
        'dvars_vstd': {
            'LongName': 'Derivative of RMS Variance over Voxels, Standardized',
            'Description': (
                'Indexes the rate of change of BOLD signal across'
                'the entire brain at each frame of data, normalized across'
                'time by that voxel standard deviation across time,'
                'before computing the RMS of the temporal difference'
            ),
        },
        'framewise_displacement': {
            'LongName': 'Framewise Displacement',
            'Description': (
                'A quantification of the estimated bulk-head'
                'motion calculated using formula proposed by Power (2012)'
            ),
            'Units': 'mm',
        },
        'aqi': {
            'LongName': "AFNI's Quality Index",
            'Description': "Mean quality index as computed by AFNI's 3dTqual",
        },
        'aor': {
            'LongName': "AFNI's Fraction of Outliers per Volume",
            'Description': (
                "Mean fraction of outliers per fMRI volume as given by AFNI's 3dToutcount"
            ),
        },
    }


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


def select_echo(
    in_files: str | list[str],
    te_echos: list[float | Undefined | None] | None = None,
    te_reference: float = 0.030,
) -> str:
    """
    Select the echo file with the closest echo time to the reference echo time.

    Used to grab the echo file when processing multi-echo data through workflows
    that only accept a single file.

    Parameters
    ----------
    in_files : :obj:`str` or :obj:`list`
        A single filename or a list of filenames.
    te_echos : :obj:`list` of :obj:`float`
        List of echo times corresponding to each file.
        If not a number (typically, a :obj:`~nipype.interfaces.base.Undefined`),
        the function selects the second echo.
    te_reference : float, optional
        Reference echo time used to find the closest echo time.

    Returns
    -------
    str
        The selected echo file.

    Examples
    --------
    >>> select_echo("single-echo.nii.gz")
    ('single-echo.nii.gz', -1)

    >>> select_echo(["single-echo.nii.gz"])
    ('single-echo.nii.gz', -1)

    >>> select_echo(
    ...     [f"echo{n}.nii.gz" for n in range(1,7)],
    ... )
    ('echo2.nii.gz', 1)

    >>> select_echo(
    ...     [f"echo{n}.nii.gz" for n in range(1,7)],
    ...     te_echos=[12.5, 28.5, 34.2, 45.0, 56.1, 68.4],
    ...     te_reference=33.1,
    ... )
    ('echo3.nii.gz', 2)

    >>> select_echo(
    ...     [f"echo{n}.nii.gz" for n in range(1,7)],
    ...     te_echos=[12.5, 28.5, 34.2, 45.0, 56.1],
    ...     te_reference=33.1,
    ... )
    ('echo2.nii.gz', 1)

    >>> select_echo(
    ...     [f"echo{n}.nii.gz" for n in range(1,7)],
    ...     te_echos=[12.5, 28.5, 34.2, 45.0, 56.1, None],
    ...     te_reference=33.1,
    ... )
    ('echo2.nii.gz', 1)

    """
    if not isinstance(in_files, (list, tuple)):
        return in_files, -1

    if len(in_files) == 1:
        return in_files[0], -1

    import numpy as np

    n_echos = len(in_files)
    if te_echos is not None and len(te_echos) == n_echos:
        try:
            index = np.argmin(np.abs(np.array(te_echos) - te_reference))
            return in_files[index], index
        except TypeError:
            pass

    return in_files[1], 1


def _get_echotime(inlist):
    if isinstance(inlist, list):
        retval = [_get_echotime(el) for el in inlist]
        return retval[0] if len(retval) == 1 else retval

    echo_time = inlist.get('EchoTime', None) if inlist else None

    if echo_time:
        return float(echo_time)
