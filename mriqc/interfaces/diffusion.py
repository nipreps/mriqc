# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# Copyright 2023 The NiPreps Developers <nipreps@gmail.com>
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
"""Interfaces for manipulating DWI data."""

from __future__ import annotations

import nibabel as nb
import numpy as np
import scipy.ndimage as nd
from dipy.core.gradients import gradient_table
from dipy.stats.qc import find_qspace_neighbors
from nipype.interfaces.base import (
    BaseInterfaceInputSpec as _BaseInterfaceInputSpec,
)
from nipype.interfaces.base import (
    File,
    InputMultiObject,
    OutputMultiObject,
    SimpleInterface,
    TraitedSpec,
    isdefined,
    traits,
)
from nipype.interfaces.base import (
    TraitedSpec as _TraitedSpec,
)
from nipype.utils.filemanip import fname_presuffix
from niworkflows.interfaces.bids import ReadSidecarJSON, _ReadSidecarJSONOutputSpec
from sklearn.cluster import KMeans
from sklearn.model_selection import GridSearchCV

from mriqc.utils.misc import _flatten_dict

__all__ = (
    'CCSegmentation',
    'CorrectSignalDrift',
    'DiffusionModel',
    'DiffusionQC',
    'ExtractOrientations',
    'FilterShells',
    'NumberOfShells',
    'PIESNO',
    'ReadDWIMetadata',
    'RotateVectors',
    'SpikingVoxelsMask',
    'SplitShells',
    'WeightedStat',
)


FD_THRESHOLD = 0.2


class _DiffusionQCInputSpec(_BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True, desc='original EPI 4D file')
    in_b0 = File(exists=True, mandatory=True, desc='input b=0 average')
    in_shells = InputMultiObject(
        File(exists=True),
        mandatory=True,
        desc='DWI data after HMC and split by shells (indexed by in_bval)',
    )
    in_shells_bval = traits.List(
        traits.Float,
        minlen=1,
        mandatory=True,
        desc='list of unique b-values (one per shell), ordered by growing intensity',
    )
    in_bval_file = File(exists=True, mandatory=True, desc='original b-vals file')
    in_bvec = traits.List(
        traits.List(
            traits.Tuple(traits.Float, traits.Float, traits.Float),
            minlen=1,
        ),
        mandatory=True,
        minlen=1,
        desc='a list of shell-wise splits of b-vectors lists -- first list are b=0',
    )
    in_bvec_rotated = traits.List(
        traits.Tuple(traits.Float, traits.Float, traits.Float),
        mandatory=True,
        minlen=1,
        desc='b-vectors after rotating by the head-motion correction transform',
    )
    in_bvec_diff = traits.List(
        traits.Float,
        mandatory=True,
        minlen=1,
        desc='list of angle deviations from the original b-vectors table',
    )
    in_fa = File(exists=True, mandatory=True, desc='input FA map')
    in_fa_nans = File(
        exists=True, mandatory=True, desc='binary mask of NaN values in the "raw" FA map'
    )
    in_fa_degenerate = File(
        exists=True,
        mandatory=True,
        desc='binary mask of values outside [0, 1] in the "raw" FA map',
    )
    in_cfa = File(exists=True, mandatory=True, desc='output color FA file')
    in_md = File(exists=True, mandatory=True, desc='input MD map')
    brain_mask = File(exists=True, mandatory=True, desc='input probabilistic brain mask')
    wm_mask = File(exists=True, mandatory=True, desc='input probabilistic white-matter mask')
    cc_mask = File(exists=True, mandatory=True, desc='input binary mask of the corpus callosum')
    spikes_mask = File(exists=True, mandatory=True, desc='input binary mask of spiking voxels')
    noise_floor = traits.Float(mandatory=True, desc='noise-floor map estimated by means of PCA')
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
    fd_thres = traits.Float(
        FD_THRESHOLD,
        usedefault=True,
        desc='FD threshold for orientation exclusion based on head motion',
    )
    in_fwhm = traits.List(traits.Float, desc='smoothness estimated with AFNI')
    qspace_neighbors = traits.List(
        traits.Tuple(traits.Int, traits.Int),
        mandatory=True,
        minlen=1,
        desc='q-space nearest neighbor pairs',
    )
    piesno_sigma = traits.Float(-1.0, usedefault=True, desc='noise sigma calculated with PIESNO')


class _DiffusionQCOutputSpec(TraitedSpec):
    bdiffs = traits.Dict
    efc = traits.Dict
    fa_degenerate = traits.Float
    fa_nans = traits.Float
    fber = traits.Dict
    fd = traits.Dict
    ndc = traits.Float
    sigma = traits.Dict
    spikes = traits.Dict
    # gsr = traits.Dict
    # tsnr = traits.Float
    # fwhm = traits.Dict(desc='full width half-maximum measure')
    # size = traits.Dict
    snr_cc = traits.Dict
    summary = traits.Dict

    out_qc = traits.Dict(desc='output flattened dictionary with all measures')


class DiffusionQC(SimpleInterface):
    """Computes :abbr:`QC (Quality Control)` measures on the input DWI EPI scan."""

    input_spec = _DiffusionQCInputSpec
    output_spec = _DiffusionQCOutputSpec

    def _run_interface(self, runtime):
        from mriqc.qc import anatomical as aqc
        from mriqc.qc import diffusion as dqc
        # from mriqc.qc import functional as fqc

        # Get the mean EPI data and get it ready
        b0nii = nb.load(self.inputs.in_b0)
        b0data = np.round(
            np.nan_to_num(np.asanyarray(b0nii.dataobj)),
            3,
        )
        b0data[b0data < 0] = 0

        # Get the FA data and get it ready OE: enable when used
        # fanii = nb.load(self.inputs.in_fa)
        # fadata = np.round(
        #     np.nan_to_num(np.asanyarray(fanii.dataobj)),
        #     3,
        # )

        # Get brain mask data
        msknii = nb.load(self.inputs.brain_mask)
        mskdata = np.round(  # Protect the thresholding with a rounding for stability
            msknii.get_fdata(),
            3,
        )
        if np.sum(mskdata) < 100:
            raise RuntimeError(
                'Detected less than 100 voxels belonging to the brain mask. '
                'MRIQC failed to process this dataset.'
            )

        # Get wm mask data
        wmnii = nb.load(self.inputs.wm_mask)
        wmdata = np.round(  # Protect the thresholding with a rounding for stability
            np.asanyarray(wmnii.dataobj),
            3,
        )

        # Get cc mask data
        ccnii = nb.load(self.inputs.cc_mask)
        ccdata = np.round(  # Protect the thresholding with a rounding for stability
            np.asanyarray(ccnii.dataobj),
            3,
        )

        # Get DWI data after splitting them by shell (DSI's data is clustered)
        shelldata = [
            np.round(
                np.asanyarray(nb.load(s).dataobj),
                4,
            )
            for s in self.inputs.in_shells
        ]

        # Summary stats
        rois = {
            'fg': mskdata,
            'bg': 1.0 - mskdata,
            'wm': wmdata,
        }
        stats = aqc.summary_stats(b0data, rois)
        self._results['summary'] = stats

        # CC mask SNR and std
        self._results['snr_cc'], cc_sigma = dqc.cc_snr(
            in_b0=b0data,
            dwi_shells=shelldata,
            cc_mask=ccdata,
            b_values=self.inputs.in_shells_bval,
            b_vectors=self.inputs.in_bvec,
        )

        fa_nans_mask = np.asanyarray(nb.load(self.inputs.in_fa_nans).dataobj) > 0.0
        self._results['fa_nans'] = round(float(1e6 * fa_nans_mask[mskdata > 0.5].mean()), 2)

        fa_degenerate_mask = np.asanyarray(nb.load(self.inputs.in_fa_degenerate).dataobj) > 0.0
        self._results['fa_degenerate'] = round(
            float(1e6 * fa_degenerate_mask[mskdata > 0.5].mean()),
            2,
        )

        # Get spikes-mask data
        spmask = np.asanyarray(nb.load(self.inputs.spikes_mask).dataobj) > 0.0
        self._results['spikes'] = dqc.spike_ppm(spmask)

        # FBER
        self._results['fber'] = {
            f'shell{i + 1:02d}': aqc.fber(bdata, mskdata.astype(np.uint8))
            for i, bdata in enumerate(shelldata)
        }

        # EFC
        self._results['efc'] = {
            f'shell{i + 1:02d}': aqc.efc(bdata) for i, bdata in enumerate(shelldata)
        }

        # FD
        fd_data = np.loadtxt(self.inputs.in_fd, skiprows=1)
        num_fd = (fd_data > self.inputs.fd_thres).sum()
        self._results['fd'] = {
            'mean': round(float(fd_data.mean()), 4),
            'num': int(num_fd),
            'perc': float(num_fd * 100 / (len(fd_data) + 1)),
        }

        # NDC
        dwidata = np.round(
            np.nan_to_num(nb.load(self.inputs.in_file).get_fdata()),
            3,
        )
        self._results['ndc'] = dqc.neighboring_dwi_correlation(
            dwidata,
            neighbor_indices=self.inputs.qspace_neighbors,
            mask=mskdata > 0.5,
        )

        # Sigmas
        self._results['sigma'] = {
            'cc': round(float(cc_sigma), 4),
            'piesno': round(self.inputs.piesno_sigma, 4),
            'pca': round(self.inputs.noise_floor, 4),
        }

        # rotated b-vecs deviations
        diffs = np.array(self.inputs.in_bvec_diff)
        self._results['bdiffs'] = {
            'mean': round(float(diffs[diffs > 1e-4].mean()), 4),
            'median': round(float(np.median(diffs[diffs > 1e-4])), 4),
            'max': round(float(diffs[diffs > 1e-4].max()), 4),
            'min': round(float(diffs[diffs > 1e-4].min()), 4),
        }

        self._results['out_qc'] = _flatten_dict(self._results)
        return runtime


class _ReadDWIMetadataOutputSpec(_ReadSidecarJSONOutputSpec):
    out_bvec_file = File(desc='corresponding bvec file')
    out_bval_file = File(desc='corresponding bval file')
    out_bmatrix = traits.List(traits.List(traits.Float), desc='b-matrix')
    qspace_neighbors = traits.List(
        traits.Tuple(traits.Int, traits.Int),
        desc='q-space nearest neighbor pairs',
    )


class ReadDWIMetadata(ReadSidecarJSON):
    """
    Extends the NiWorkflows' interface to extract bvec/bval from DWI datasets.
    """

    output_spec = _ReadDWIMetadataOutputSpec

    def _run_interface(self, runtime):
        runtime = super()._run_interface(runtime)

        self._results['out_bvec_file'] = str(self.layout.get_bvec(self.inputs.in_file))
        self._results['out_bval_file'] = str(self.layout.get_bval(self.inputs.in_file))

        bvecs = np.loadtxt(self._results['out_bvec_file']).T
        bvals = np.loadtxt(self._results['out_bval_file'])

        gtab = gradient_table(bvals, bvecs=bvecs)

        self._results['qspace_neighbors'] = find_qspace_neighbors(gtab)
        self._results['out_bmatrix'] = np.hstack((bvecs, bvals[:, np.newaxis])).tolist()

        return runtime


class _WeightedStatInputSpec(_BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True, desc='an image')
    in_weights = traits.List(
        traits.Either(traits.Bool, traits.Float),
        mandatory=True,
        minlen=1,
        desc='list of weights',
    )
    stat = traits.Enum('mean', 'std', usedefault=True, desc='statistic to compute')


class _WeightedStatOutputSpec(_TraitedSpec):
    out_file = File(exists=True, desc='masked file')


class WeightedStat(SimpleInterface):
    """Weighted average of the input image across the last dimension."""

    input_spec = _WeightedStatInputSpec
    output_spec = _WeightedStatOutputSpec

    def _run_interface(self, runtime):
        img = nb.load(self.inputs.in_file)
        weights = [float(w) for w in self.inputs.in_weights]
        data = np.asanyarray(img.dataobj)
        statmap = np.average(data, weights=weights, axis=-1)

        self._results['out_file'] = fname_presuffix(
            self.inputs.in_file, suffix=f'_{self.inputs.stat}', newpath=runtime.cwd
        )

        if self.inputs.stat == 'std':
            statmap = np.sqrt(
                np.average((data - statmap[..., np.newaxis]) ** 2, weights=weights, axis=-1)
            )

        hdr = img.header.copy()
        img.__class__(
            statmap.astype(hdr.get_data_dtype()),
            img.affine,
            hdr,
        ).to_filename(self._results['out_file'])

        return runtime


class _NumberOfShellsInputSpec(_BaseInterfaceInputSpec):
    in_bvals = File(mandatory=True, desc='bvals file')
    b0_threshold = traits.Float(50, usedefault=True, desc='a threshold for the low-b values')
    dsi_threshold = traits.Int(11, usedefault=True, desc='number of shells to call a dataset DSI')


class _NumberOfShellsOutputSpec(_TraitedSpec):
    models = traits.List(traits.Int, minlen=1, desc='number of shells ordered by model fit')
    n_shells = traits.Int(desc='number of shells')
    out_data = traits.List(
        traits.Float,
        minlen=1,
        desc="list of new b-values (e.g., after 'shell-ifying' DSI)",
    )
    b_values = traits.List(
        traits.Float,
        minlen=1,
        desc='list of ``n_shells`` b-values associated with each shell (only nonzero)',
    )
    b_masks = traits.List(
        traits.List(traits.Bool, minlen=1),
        minlen=1,
        desc='list of ``n_shells`` b-value-wise masks',
    )
    b_indices = traits.List(
        traits.List(traits.Int, minlen=1),
        minlen=1,
        desc='list of ``n_shells`` b-value-wise indices lists',
    )
    b_dict = traits.Dict(
        traits.Int, traits.List(traits.Int), desc='a map of b-values (including b=0) and masks'
    )


class NumberOfShells(SimpleInterface):
    """
    Weighted average of the input image across the last dimension.

    Examples
    --------
    >>> np.savetxt("test.bval", [0] * 8 + [1000] * 12 + [2000] * 10)
    >>> NumberOfShells(in_bvals="test.bval").run().outputs.n_shells
    2
    >>> np.savetxt("test.bval", [0] * 8 + [1000] * 12)
    >>> NumberOfShells(in_bvals="test.bval").run().outputs.n_shells
    1
    >>> np.savetxt("test.bval", np.arange(0, 9001, 120))
    >>> NumberOfShells(in_bvals="test.bval").run().outputs.n_shells > 7
    True

    """

    input_spec = _NumberOfShellsInputSpec
    output_spec = _NumberOfShellsOutputSpec

    def _run_interface(self, runtime):
        in_data = np.squeeze(np.loadtxt(self.inputs.in_bvals))
        highb_mask = in_data > self.inputs.b0_threshold

        original_bvals = sorted(set(np.rint(in_data[highb_mask]).astype(int)))
        round_bvals = np.round(in_data, -2).astype(int)
        shell_bvals = sorted(set(round_bvals[highb_mask]))

        if len(shell_bvals) <= self.inputs.dsi_threshold:
            self._results['n_shells'] = len(shell_bvals)
            self._results['models'] = [self._results['n_shells']]
            self._results['out_data'] = round_bvals.tolist()
            self._results['b_values'] = shell_bvals
        else:
            # For datasets identified as DSI, fit a k-means
            grid_search = GridSearchCV(
                KMeans(), param_grid={'n_clusters': range(1, 10)}, scoring=_rms
            ).fit(in_data[highb_mask].reshape(-1, 1))

            results = np.array(
                sorted(
                    zip(
                        grid_search.cv_results_['mean_test_score'] * -1.0,
                        grid_search.cv_results_['param_n_clusters'],
                    )
                )
            )

            self._results['models'] = results[:, 1].astype(int).tolist()
            self._results['n_shells'] = int(grid_search.best_params_['n_clusters'])

            out_data = np.zeros_like(in_data)
            predicted_shell = np.rint(
                np.squeeze(
                    grid_search.best_estimator_.cluster_centers_[
                        grid_search.best_estimator_.predict(in_data[highb_mask].reshape(-1, 1))
                    ],
                )
            ).astype(int)

            # If estimated shells matches direct count, probably right -- do not change b-vals
            if len(original_bvals) == self._results['n_shells']:
                # Find closest b-values
                indices = np.abs(predicted_shell[:, np.newaxis] - original_bvals).argmin(axis=1)
                predicted_shell = original_bvals[indices]

            out_data[highb_mask] = predicted_shell
            self._results['out_data'] = np.round(out_data.astype(float), 2).tolist()
            self._results['b_values'] = sorted(
                np.unique(np.round(predicted_shell.astype(float), 2)).tolist()
            )

        self._results['b_masks'] = [(~highb_mask).tolist()] + [
            np.isclose(self._results['out_data'], bvalue).tolist()
            for bvalue in self._results['b_values']
        ]
        self._results['b_indices'] = [
            np.atleast_1d(np.squeeze(np.argwhere(b_mask)).astype(int)).tolist()
            for b_mask in self._results['b_masks']
        ]

        self._results['b_dict'] = {
            int(round(k, 0)): value
            for k, value in zip([0] + self._results['b_values'], self._results['b_indices'])
        }
        return runtime


class _ExtractOrientationsInputSpec(_BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True, desc='dwi file')
    indices = traits.List(traits.Int, mandatory=True, desc='indices to be extracted')
    in_bvec_file = File(exists=True, desc='b-vectors file')


class _ExtractOrientationsOutputSpec(_TraitedSpec):
    out_file = File(exists=True, desc='output b0 file')
    out_bvec = traits.List(
        traits.Tuple(traits.Float, traits.Float, traits.Float),
        minlen=1,
        desc='b-vectors',
    )


class ExtractOrientations(SimpleInterface):
    """Extract all b=0 volumes from a dwi series."""

    input_spec = _ExtractOrientationsInputSpec
    output_spec = _ExtractOrientationsOutputSpec

    def _run_interface(self, runtime):
        from nipype.utils.filemanip import fname_presuffix

        out_file = fname_presuffix(
            self.inputs.in_file,
            suffix='_subset',
            newpath=runtime.cwd,
        )

        self._results['out_file'] = out_file

        img = nb.load(self.inputs.in_file)
        bzeros = np.squeeze(np.asanyarray(img.dataobj)[..., self.inputs.indices])

        hdr = img.header.copy()
        hdr.set_data_shape(bzeros.shape)
        hdr.set_xyzt_units('mm')
        nb.Nifti1Image(bzeros, img.affine, hdr).to_filename(out_file)

        if isdefined(self.inputs.in_bvec_file):
            bvecs = np.loadtxt(self.inputs.in_bvec_file)[:, self.inputs.indices].T
            self._results['out_bvec'] = [tuple(row) for row in bvecs]

        return runtime


class _CorrectSignalDriftInputSpec(_BaseInterfaceInputSpec):
    in_file = File(
        exists=True,
        mandatory=True,
        desc='a 4D file with (exclusively) realigned low-b volumes',
    )
    bias_file = File(exists=True, desc='a B1 bias field')
    brainmask_file = File(exists=True, desc='a 3D file of the brain mask')
    b0_ixs = traits.List(traits.Int, mandatory=True, desc='Index of b0s')
    bval_file = File(exists=True, mandatory=True, desc='bvalues file')
    full_epi = File(exists=True, desc='a whole DWI dataset to be corrected for drift')


class _CorrectSignalDriftOutputSpec(_TraitedSpec):
    out_file = File(desc='a 4D file with (exclusively) realigned, drift-corrected low-b volumes')
    out_full_file = File(desc='full DWI input after drift correction')
    b0_drift = traits.List(traits.Float, desc='global signal evolution')
    signal_drift = traits.List(traits.Float, desc='signal drift after fiting exp decay')


class CorrectSignalDrift(SimpleInterface):
    """Correct DWI for signal drift."""

    input_spec = _CorrectSignalDriftInputSpec
    output_spec = _CorrectSignalDriftOutputSpec

    def _run_interface(self, runtime):
        from mriqc import config

        bvals = np.loadtxt(self.inputs.bval_file)
        len_dmri = bvals.size

        img = nb.load(self.inputs.in_file)
        data = img.get_fdata()
        bmask = np.ones_like(data[..., 0], dtype=bool)

        # Correct for the B1 bias
        if isdefined(self.inputs.bias_file):
            data *= nb.load(self.inputs.bias_file).get_fdata()[..., np.newaxis]

        if isdefined(self.inputs.brainmask_file):
            bmask = np.round(nb.load(self.inputs.brainmask_file).get_fdata(), 2) > 0.5

        self._results['out_file'] = fname_presuffix(
            self.inputs.in_file, suffix='_nodrift', newpath=runtime.cwd
        )

        if (b0len := int(data.ndim < 4)) or (b0len := data.shape[3]) < 3:
            config.loggers.interface.warn(
                f'Insufficient number of low-b orientations ({b0len}) '
                'to safely calculate signal drift.'
            )

            img.__class__(
                np.round(data.astype('float32'), 4),
                img.affine,
                img.header,
            ).to_filename(self._results['out_file'])

            if isdefined(self.inputs.full_epi):
                self._results['out_full_file'] = self.inputs.full_epi

            self._results['b0_drift'] = [1.0] * b0len
            self._results['signal_drift'] = [1.0] * len_dmri

            return runtime

        global_signal = np.array(
            [np.median(data[..., n_b0][bmask]) for n_b0 in range(img.shape[-1])]
        ).astype('float32')

        # Normalize and correct
        global_signal /= global_signal[0]
        self._results['b0_drift'] = [round(float(gs), 4) for gs in global_signal]

        config.loggers.interface.info(
            f'Correcting drift with {len(global_signal)} b=0 volumes, with '
            'global signal estimated at '
            f'{", ".join([str(v) for v in self._results["b0_drift"]])}.'
        )

        data *= 1.0 / global_signal[np.newaxis, np.newaxis, np.newaxis, :]

        img.__class__(
            data.astype(img.header.get_data_dtype()),
            img.affine,
            img.header,
        ).to_filename(self._results['out_file'])

        # Fit line to log-transformed drifts
        K, A_log = np.polyfit(self.inputs.b0_ixs, np.log(global_signal), 1)

        t_points = np.arange(len_dmri, dtype=int)
        fitted = np.squeeze(_exp_func(t_points, np.exp(A_log), K, 0))
        self._results['signal_drift'] = fitted.astype(float).tolist()

        if isdefined(self.inputs.full_epi):
            self._results['out_full_file'] = fname_presuffix(
                self.inputs.full_epi, suffix='_nodriftfull', newpath=runtime.cwd
            )
            full_img = nb.load(self.inputs.full_epi)
            full_img.__class__(
                full_img.get_fdata() * fitted[np.newaxis, np.newaxis, np.newaxis, :],
                full_img.affine,
                full_img.header,
            ).to_filename(self._results['out_full_file'])
        return runtime


class _SplitShellsInputSpec(_BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True, desc='dwi file')
    bvals = traits.List(traits.Float, mandatory=True, desc='bval table')


class _SplitShellsOutputSpec(_TraitedSpec):
    out_file = OutputMultiObject(File(exists=True), desc='output b0 file')


class SplitShells(SimpleInterface):
    """Split a DWI dataset into ."""

    input_spec = _SplitShellsInputSpec
    output_spec = _SplitShellsOutputSpec

    def _run_interface(self, runtime):
        from nipype.utils.filemanip import fname_presuffix

        bval_list = np.rint(self.inputs.bvals).astype(int)
        bvals = np.unique(bval_list)
        img = nb.load(self.inputs.in_file)
        data = np.asanyarray(img.dataobj)

        self._results['out_file'] = []

        for bval in bvals:
            fname = fname_presuffix(
                self.inputs.in_file, suffix=f'_b{bval:05d}', newpath=runtime.cwd
            )
            self._results['out_file'].append(fname)

            img.__class__(
                data[..., np.argwhere(bval_list == bval)],
                img.affine,
                img.header,
            ).to_filename(fname)
        return runtime


class _FilterShellsInputSpec(_BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True, desc='dwi file')
    bvals = traits.List(traits.Float, mandatory=True, desc='bval table')
    bvec_file = File(exists=True, mandatory=True, desc='b-vectors')
    b_threshold = traits.Float(1100, usedefault=True, desc='b-values threshold')


class _FilterShellsOutputSpec(_TraitedSpec):
    out_file = File(exists=True, desc='filtered DWI file')
    out_bvals = traits.List(traits.Float, desc='filtered bvalues')
    out_bvec_file = File(exists=True, desc='filtered bvecs file')
    out_bval_file = File(exists=True, desc='filtered bvals file')


class FilterShells(SimpleInterface):
    """Extract DWIs below a given b-value threshold."""

    input_spec = _FilterShellsInputSpec
    output_spec = _FilterShellsOutputSpec

    def _run_interface(self, runtime):
        from nipype.utils.filemanip import fname_presuffix

        bvals = np.array(self.inputs.bvals)
        bval_mask = bvals < self.inputs.b_threshold
        bvecs = np.loadtxt(self.inputs.bvec_file)[:, bval_mask]

        self._results['out_bvals'] = bvals[bval_mask].astype(float).tolist()
        self._results['out_bvec_file'] = fname_presuffix(
            self.inputs.in_file,
            suffix='_dti.bvec',
            newpath=runtime.cwd,
            use_ext=False,
        )
        np.savetxt(self._results['out_bvec_file'], bvecs)

        self._results['out_bval_file'] = fname_presuffix(
            self.inputs.in_file,
            suffix='_dti.bval',
            newpath=runtime.cwd,
            use_ext=False,
        )
        np.savetxt(self._results['out_bval_file'], bvals)

        self._results['out_file'] = fname_presuffix(
            self.inputs.in_file,
            suffix='_dti',
            newpath=runtime.cwd,
        )

        dwi_img = nb.load(self.inputs.in_file)
        data = np.array(dwi_img.dataobj, dtype=dwi_img.header.get_data_dtype())[..., bval_mask]
        dwi_img.__class__(
            data,
            dwi_img.affine,
            dwi_img.header,
        ).to_filename(self._results['out_file'])

        return runtime


class _DiffusionModelInputSpec(_BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True, desc='dwi file')
    bvals = traits.List(traits.Float, mandatory=True, desc='bval table')
    bvec_file = File(exists=True, mandatory=True, desc='b-vectors')
    brain_mask = File(exists=True, desc='brain mask file')
    decimals = traits.Int(3, usedefault=True, desc='round output maps for reliability')
    n_shells = traits.Int(mandatory=True, desc='number of shells')


class _DiffusionModelOutputSpec(_TraitedSpec):
    out_fa = File(exists=True, desc='output FA file')
    out_fa_nans = File(exists=True, desc='binary mask of NaN values in the "raw" FA map')
    out_fa_degenerate = File(
        exists=True,
        desc='binary mask of values outside [0, 1] in the "raw" FA map',
    )
    out_cfa = File(exists=True, desc='output color FA file')
    out_md = File(exists=True, desc='output MD file')


class DiffusionModel(SimpleInterface):
    """
    Fit a :obj:`~dipy.reconst.dki.DiffusionKurtosisModel` on the dataset.

    If ``n_shells`` is set to 1, then a :obj:`~dipy.reconst.dti.TensorModel`
    is used.

    """

    input_spec = _DiffusionModelInputSpec
    output_spec = _DiffusionModelOutputSpec

    def _run_interface(self, runtime):
        from dipy.core.gradients import gradient_table_from_bvals_bvecs
        from nipype.utils.filemanip import fname_presuffix

        bvals = np.array(self.inputs.bvals)

        gtab = gradient_table_from_bvals_bvecs(
            bvals=bvals,
            bvecs=np.loadtxt(self.inputs.bvec_file).T,
        )

        img = nb.load(self.inputs.in_file)
        data = img.get_fdata(dtype='float32')

        brainmask = np.ones_like(data[..., 0], dtype=bool)

        if isdefined(self.inputs.brain_mask):
            brainmask = (
                np.round(
                    nb.load(self.inputs.brain_mask).get_fdata(),
                    3,
                )
                > 0.5
            )

        if self.inputs.n_shells == 1:
            from dipy.reconst.dti import TensorModel as Model
        else:
            from dipy.reconst.dki import DiffusionKurtosisModel as Model

        # Fit DIT
        fwdtifit = Model(gtab).fit(
            data,
            mask=brainmask,
        )

        # Extract the FA
        fa_data = fwdtifit.fa
        fa_nan_msk = np.isnan(fa_data)
        fa_data[fa_nan_msk] = 0

        # Round for stability
        fa_data = np.round(fa_data, self.inputs.decimals)
        degenerate_msk = (fa_data < 0) | (fa_data > 1.0)
        # Clamp the FA to remove degenerate
        fa_data = np.clip(fa_data, 0, 1)

        fa_nii = nb.Nifti1Image(
            fa_data,
            img.affine,
            None,
        )

        fa_nii.header.set_xyzt_units('mm')
        fa_nii.header.set_intent('estimate', name='Fractional Anisotropy (FA)')

        self._results['out_fa'] = fname_presuffix(
            self.inputs.in_file,
            suffix='fa',
            newpath=runtime.cwd,
        )

        fa_nii.to_filename(self._results['out_fa'])

        # Write out degenerate and nans masks
        fa_nan_nii = nb.Nifti1Image(
            fa_nan_msk.astype(np.uint8),
            img.affine,
            None,
        )

        fa_nan_nii.header.set_xyzt_units('mm')
        fa_nan_nii.header.set_intent('estimate', name='NaNs in the FA map mask')
        fa_nan_nii.header['cal_max'] = 1
        fa_nan_nii.header['cal_min'] = 0

        self._results['out_fa_nans'] = fname_presuffix(
            self.inputs.in_file,
            suffix='desc-fanans_mask',
            newpath=runtime.cwd,
        )
        fa_nan_nii.to_filename(self._results['out_fa_nans'])

        fa_degenerate_nii = nb.Nifti1Image(
            degenerate_msk.astype(np.uint8),
            img.affine,
            None,
        )

        fa_degenerate_nii.header.set_xyzt_units('mm')
        fa_degenerate_nii.header.set_intent(
            'estimate', name='degenerate vectors in the FA map mask'
        )
        fa_degenerate_nii.header['cal_max'] = 1
        fa_degenerate_nii.header['cal_min'] = 0

        self._results['out_fa_degenerate'] = fname_presuffix(
            self.inputs.in_file,
            suffix='desc-fadegenerate_mask',
            newpath=runtime.cwd,
        )
        fa_degenerate_nii.to_filename(self._results['out_fa_degenerate'])

        # Extract the color FA
        cfa_data = fwdtifit.color_fa
        cfa_nii = nb.Nifti1Image(
            np.clip(cfa_data, a_min=0.0, a_max=1.0),
            img.affine,
            None,
        )

        cfa_nii.header.set_xyzt_units('mm')
        cfa_nii.header.set_intent('estimate', name='Fractional Anisotropy (FA)')
        cfa_nii.header['cal_max'] = 1.0
        cfa_nii.header['cal_min'] = 0.0

        self._results['out_cfa'] = fname_presuffix(
            self.inputs.in_file,
            suffix='cfa',
            newpath=runtime.cwd,
        )
        cfa_nii.to_filename(self._results['out_cfa'])

        # Extract the AD
        self._results['out_md'] = fname_presuffix(
            self.inputs.in_file,
            suffix='md',
            newpath=runtime.cwd,
        )
        md_data = np.array(fwdtifit.md, dtype='float32')
        md_data[np.isnan(md_data)] = 0
        md_data = np.clip(md_data, 0, 1)
        md_hdr = fa_nii.header.copy()
        md_hdr.set_intent('estimate', name='Mean diffusivity (MD)')
        nb.Nifti1Image(md_data, img.affine, md_hdr).to_filename(self._results['out_md'])

        return runtime


class _CCSegmentationInputSpec(_BaseInterfaceInputSpec):
    in_fa = File(exists=True, mandatory=True, desc='fractional anisotropy (FA) file')
    in_cfa = File(exists=True, mandatory=True, desc='color FA file')
    min_rgb = traits.Tuple(
        (0.4, 0.008, 0.008),
        types=(traits.Float,) * 3,
        usedefault=True,
        desc='minimum RGB within the CC',
    )
    max_rgb = traits.Tuple(
        (1.1, 0.25, 0.25),
        types=(traits.Float,) * 3,
        usedefault=True,
        desc='maximum RGB within the CC',
    )
    wm_threshold = traits.Float(0.35, usedefault=True, desc='WM segmentation threshold')
    clean_mask = traits.Bool(False, usedefault=True, desc='run a final cleanup step on mask')


class _CCSegmentationOutputSpec(_TraitedSpec):
    out_mask = File(exists=True, desc='output mask of the corpus callosum')
    wm_mask = File(exists=True, desc='output mask of the white-matter (thresholded)')
    wm_finalmask = File(exists=True, desc='output mask of the white-matter after binary opening')


class CCSegmentation(SimpleInterface):
    """Computes :abbr:`QC (Quality Control)` measures on the input DWI EPI scan."""

    input_spec = _CCSegmentationInputSpec
    output_spec = _CCSegmentationOutputSpec

    def _run_interface(self, runtime):
        from skimage.measure import label

        self._results['out_mask'] = fname_presuffix(
            self.inputs.in_cfa,
            suffix='ccmask',
            newpath=runtime.cwd,
        )
        self._results['wm_mask'] = fname_presuffix(
            self.inputs.in_cfa,
            suffix='wmmask',
            newpath=runtime.cwd,
        )
        self._results['wm_finalmask'] = fname_presuffix(
            self.inputs.in_cfa,
            suffix='wmfinalmask',
            newpath=runtime.cwd,
        )

        fa_nii = nb.load(self.inputs.in_fa)
        fa_data = np.round(fa_nii.get_fdata(dtype='float32'), 4)
        fa_labels = label((fa_data > self.inputs.wm_threshold).astype(np.uint8))
        wm_mask = fa_labels == np.argmax(np.bincount(fa_labels.flat)[1:]) + 1

        # Write out binary WM mask
        wm_mask_nii = nb.Nifti1Image(
            wm_mask.astype(np.uint8),
            fa_nii.affine,
            None,
        )
        wm_mask_nii.header.set_xyzt_units('mm')
        wm_mask_nii.header.set_intent('estimate', name='white-matter mask (FA thresholded)')
        wm_mask_nii.header['cal_max'] = 1
        wm_mask_nii.header['cal_min'] = 0
        wm_mask_nii.to_filename(self._results['wm_mask'])

        # Massage FA with greyscale mathematical morphology
        struct = nd.generate_binary_structure(wm_mask.ndim, wm_mask.ndim - 1)
        # Perform a closing followed by opening operations on the FA.
        wm_mask = nd.grey_closing(
            fa_data,
            structure=struct,
        )
        wm_mask = nd.grey_opening(
            wm_mask,
            structure=struct,
        )

        fa_labels = label((np.round(wm_mask, 4) > self.inputs.wm_threshold).astype(np.uint8))
        wm_mask = fa_labels == np.argmax(np.bincount(fa_labels.flat)[1:]) + 1

        # Write out binary WM mask after binary opening
        wm_mask_nii = nb.Nifti1Image(
            wm_mask.astype(np.uint8),
            fa_nii.affine,
            wm_mask_nii.header,
        )
        wm_mask_nii.header.set_intent('estimate', name='white-matter mask after binary opening')
        wm_mask_nii.to_filename(self._results['wm_finalmask'])

        cfa_data = np.round(nb.load(self.inputs.in_cfa).get_fdata(dtype='float32'), 4)
        for i in range(cfa_data.shape[-1]):
            cfa_data[..., i] = nd.grey_closing(
                cfa_data[..., i],
                structure=struct,
            )
            cfa_data[..., i] = nd.grey_opening(
                cfa_data[..., i],
                structure=struct,
            )

        cc_mask = segment_corpus_callosum(
            in_cfa=cfa_data,
            mask=wm_mask,
            min_rgb=self.inputs.min_rgb,
            max_rgb=self.inputs.max_rgb,
            clean_mask=self.inputs.clean_mask,
        )
        cc_mask_nii = nb.Nifti1Image(
            cc_mask.astype(np.uint8),
            fa_nii.affine,
            None,
        )
        cc_mask_nii.header.set_xyzt_units('mm')
        cc_mask_nii.header.set_intent('estimate', name='corpus callosum mask')
        cc_mask_nii.header['cal_max'] = 1
        cc_mask_nii.header['cal_min'] = 0
        cc_mask_nii.to_filename(self._results['out_mask'])
        return runtime


class _SpikingVoxelsMaskInputSpec(_BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True, desc='a DWI 4D file')
    brain_mask = File(exists=True, mandatory=True, desc='input probabilistic brain 3D mask')
    z_threshold = traits.Float(3.0, usedefault=True, desc='z-score threshold')
    b_masks = traits.List(
        traits.List(traits.Int, minlen=1),
        minlen=1,
        mandatory=True,
        desc='list of ``n_shells`` b-value-wise indices lists',
    )


class _SpikingVoxelsMaskOutputSpec(_TraitedSpec):
    out_mask = File(exists=True, desc='a 4D binary mask of spiking voxels')


class SpikingVoxelsMask(SimpleInterface):
    """Computes :abbr:`QC (Quality Control)` measures on the input DWI EPI scan."""

    input_spec = _SpikingVoxelsMaskInputSpec
    output_spec = _SpikingVoxelsMaskOutputSpec

    def _run_interface(self, runtime):
        self._results['out_mask'] = fname_presuffix(
            self.inputs.in_file,
            suffix='spikesmask',
            newpath=runtime.cwd,
        )

        in_nii = nb.load(self.inputs.in_file)
        data = np.round(in_nii.get_fdata(), 4).astype('float32')

        bmask_nii = nb.load(self.inputs.brain_mask)
        brainmask = np.round(bmask_nii.get_fdata(), 2).astype('float32')

        spikes_mask = get_spike_mask(
            data,
            shell_masks=self.inputs.b_masks,
            brainmask=brainmask,
            z_threshold=self.inputs.z_threshold,
        )

        header = bmask_nii.header.copy()
        header.set_data_dtype(np.uint8)
        header.set_xyzt_units('mm')
        header.set_intent('estimate', name='spiking voxels mask')
        header['cal_max'] = 1
        header['cal_min'] = 0

        # Write out binary WM mask after binary opening
        spikes_mask_nii = nb.Nifti1Image(
            spikes_mask.astype(np.uint8),
            bmask_nii.affine,
            header,
        )
        spikes_mask_nii.to_filename(self._results['out_mask'])

        return runtime


class _PIESNOInputSpec(_BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True, desc='a DWI 4D file')
    n_channels = traits.Int(4, usedefault=True, min=1, desc='number of channels')


class _PIESNOOutputSpec(_TraitedSpec):
    sigma = traits.Float(desc='noise sigma calculated with PIESNO')
    out_mask = File(exists=True, desc='a 4D binary mask of spiking voxels')


class PIESNO(SimpleInterface):
    """Computes :abbr:`QC (Quality Control)` measures on the input DWI EPI scan."""

    input_spec = _PIESNOInputSpec
    output_spec = _PIESNOOutputSpec

    def _run_interface(self, runtime):
        self._results['out_mask'] = fname_presuffix(
            self.inputs.in_file,
            suffix='piesno',
            newpath=runtime.cwd,
        )

        in_nii = nb.load(self.inputs.in_file)
        data = np.round(in_nii.get_fdata(), 4).astype('float32')

        sigma, maskdata = noise_piesno(data)

        header = in_nii.header.copy()
        header.set_data_dtype(np.uint8)
        header.set_xyzt_units('mm')
        header.set_intent('estimate', name='PIESNO noise voxels mask')
        header['cal_max'] = 1
        header['cal_min'] = 0

        nb.Nifti1Image(
            maskdata.astype(np.uint8),
            in_nii.affine,
            header,
        ).to_filename(self._results['out_mask'])

        self._results['sigma'] = round(float(np.median(sigma)), 5)
        return runtime


class _RotateVectorsInputSpec(_BaseInterfaceInputSpec):
    in_file = File(
        exists=True,
        mandatory=True,
        desc='TSV file containing original b-vectors and b-values',
    )
    reference = File(
        exists=True,
        mandatory=True,
        desc='dwi-related file providing the reference affine',
    )
    transforms = File(exists=True, desc='list of head-motion transforms')


class _RotateVectorsOutputSpec(_TraitedSpec):
    out_bvec = traits.List(
        traits.Tuple(traits.Float, traits.Float, traits.Float),
        minlen=1,
        desc='rotated b-vectors',
    )
    out_diff = traits.List(
        traits.Float,
        minlen=1,
        desc='angles in radians between new b-vectors and the original ones',
    )


class RotateVectors(SimpleInterface):
    """Extract all b=0 volumes from a dwi series."""

    input_spec = _RotateVectorsInputSpec
    output_spec = _RotateVectorsOutputSpec

    def _run_interface(self, runtime):
        from nitransforms.linear import load

        vox2ras = nb.load(self.inputs.reference).affine
        ras2vox = np.linalg.inv(vox2ras)

        ijk = np.loadtxt(self.inputs.in_file).T
        nonzero = np.linalg.norm(ijk, axis=1) > 1e-3

        xyz = (vox2ras[:3, :3] @ ijk.T).T

        # Unit vectors in RAS coordinates
        xyz_norms = np.linalg.norm(xyz, axis=1)
        xyz[nonzero] = xyz[nonzero] / xyz_norms[nonzero, np.newaxis]

        hmc_rot = load(self.inputs.transforms).matrix[:, :3, :3]
        ijk_rotated = (ras2vox[:3, :3] @ np.einsum('ijk,ik->ij', hmc_rot, xyz).T).T.astype(
            'float32'
        )
        ijk_rotated_norm = np.linalg.norm(ijk_rotated, axis=1)
        ijk_rotated[nonzero] = ijk_rotated[nonzero] / ijk_rotated_norm[nonzero, np.newaxis]
        ijk_rotated[~nonzero] = ijk[~nonzero]

        self._results['out_bvec'] = list(
            zip(ijk_rotated[:, 0], ijk_rotated[:, 1], ijk_rotated[:, 2])
        )

        diffs = np.zeros_like(ijk[:, 0])
        diffs[nonzero] = np.arccos(
            np.clip(np.einsum('ij, ij->i', ijk[nonzero], ijk_rotated[nonzero]), -1.0, 1.0)
        )
        self._results['out_diff'] = [round(float(v), 6) for v in diffs]

        return runtime


def _rms(estimator, X):
    """
    Callable to pass to GridSearchCV that will calculate a distance score.

    To consider: using `MDL
    <https://erikerlandson.github.io/blog/2016/08/03/x-medoids-using-minimum-description-length-to-identify-the-k-in-k-medoids/>`__

    """
    if len(np.unique(estimator.cluster_centers_)) < estimator.n_clusters:
        return -np.inf

    # Calculate distance from assigned shell centroid
    distance = X - estimator.cluster_centers_[estimator.predict(X)]
    # Make negative so CV optimizes minimizes the error
    return -np.sqrt(distance**2).sum()


def _exp_func(t, A, K, C):
    return A * np.exp(K * t) + C


def segment_corpus_callosum(
    in_cfa: np.ndarray,
    mask: np.ndarray,
    min_rgb: tuple[float, float, float] = (0.6, 0.0, 0.0),
    max_rgb: tuple[float, float, float] = (1.0, 0.1, 0.1),
    clean_mask: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Segments the corpus callosum (CC) from a color FA map.

    Parameters
    ----------
    in_cfa : :obj:`~numpy.ndarray`
        The color FA (cFA) map.
    mask : :obj:`~numpy.ndarray` (bool, 3D)
        A white matter mask used to define the initial bounding box.
    min_rgb : :obj:`tuple`, optional
        Minimum RGB values.
    max_rgb : :obj:`tuple`, optional
        Maximum RGB values.
    clean_mask : :obj:`bool`, optional
        Whether the CC mask is finally cleaned-up for spurious off voxels with
        :obj:`dipy.segment.mask.clean_cc_mask`

    Returns
    -------
    cc_mask: :obj:`~numpy.ndarray`
        The final binary mask of the segmented CC.

    Notes
    -----
    This implementation was derived from
    :obj:`dipy.segment.mask.segment_from_cfa`.

    """
    from dipy.segment.mask import bounding_box

    # Prepare a bounding box of the CC
    cc_box = np.zeros_like(mask, dtype=bool)
    mins, maxs = bounding_box(mask)  # mask needs to be volume
    mins = np.array(mins)
    maxs = np.array(maxs)
    diff = (maxs - mins) // 5
    bounds_min = mins + diff
    bounds_max = maxs - diff
    cc_box[
        bounds_min[0] : bounds_max[0], bounds_min[1] : bounds_max[1], bounds_min[2] : bounds_max[2]
    ] = True

    min_rgb = np.array(min_rgb)
    max_rgb = np.array(max_rgb)

    # Threshold color FA
    cc_mask = np.all(
        (in_cfa >= min_rgb[None, :]) & (in_cfa <= max_rgb[None, :]),
        axis=-1,
    )

    # Apply bounding box and WM mask
    cc_mask *= cc_box & mask

    struct = nd.generate_binary_structure(cc_mask.ndim, cc_mask.ndim - 1)
    # Perform a closing followed by opening operations on the FA.
    cc_mask = nd.binary_closing(
        cc_mask,
        structure=struct,
    )
    cc_mask = nd.binary_opening(
        cc_mask,
        structure=struct,
    )

    if clean_mask:
        from dipy.segment.mask import clean_cc_mask

        cc_mask = clean_cc_mask(cc_mask)
    return cc_mask


def get_spike_mask(
    data: np.ndarray,
    shell_masks: list,
    brainmask: np.ndarray,
    z_threshold: float = 3.0,
) -> np.ndarray:
    """
    Creates a binary mask classifying voxels in the data array as spike or non-spike.

    This function identifies voxels with signal intensities exceeding a threshold based
    on standard deviations above the mean. The threshold can be applied globally to
    the entire data array, or it can be calculated for groups of voxels defined by
    the ``grouping_vals`` parameter.

    Parameters
    ----------
    data : :obj:`~numpy.ndarray`
        The data array to be thresholded.
    z_threshold : :obj:`float`, optional (default=3.0)
        The number of standard deviations to use above the mean as the threshold
        multiplier.
    brainmask : :obj:`~numpy.ndarray`
        The brain mask.
    shell_masks : :obj:`list`
        A list of :obj:`~numpy.ndarray` objects

    Returns:
    -------
    spike_mask : :obj:`~numpy.ndarray`
        A binary mask where ``True`` values indicate voxels classified as spikes and
        ``False`` values indicate non-spikes. The mask has the same shape as the input
        data array.

    """

    spike_mask = np.zeros_like(data, dtype=bool)

    brainmask = brainmask >= 0.5

    for b_mask in shell_masks:
        shelldata = data[..., b_mask]

        a_thres = z_threshold * shelldata[brainmask].std() + shelldata[brainmask].mean()

        spike_mask[..., b_mask] = shelldata > a_thres

    return spike_mask


def noise_piesno(data: np.ndarray, n_channels: int = 4) -> (np.ndarray, np.ndarray):
    """
    Estimates noise in raw diffusion MRI (dMRI) data using the PIESNO algorithm.

    This function implements the PIESNO (Probabilistic Identification and Estimation
    of Noise) algorithm [Koay2009]_ to estimate the standard deviation (sigma) of the
    noise in each voxel of a 4D dMRI data array. The PIESNO algorithm assumes Rician
    distributed signal and exploits the statistical properties of the noise to
    separate it from the underlying signal.

    Parameters
    ----------
    data : :obj:`~numpy.ndarray`
        The 4D raw dMRI data array.
    n_channels : :obj:`int`, optional (default=4)
        The number of diffusion-encoding channels in the data. This value is used
        internally by the PIESNO algorithm.

    Returns
    -------
    sigma : :obj:`~numpy.ndarray`
        The estimated noise standard deviation for each voxel in the data array.
    mask : :obj:`~numpy.ndarray`
        A brain mask estimated by PIESNO. This mask identifies voxels containing
        mostly noise and can be used for further processing.

    """
    from dipy.denoise.noise_estimate import piesno

    sigma, mask = piesno(data, N=n_channels, return_mask=True)
    return sigma, mask
