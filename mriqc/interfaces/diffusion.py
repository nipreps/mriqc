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
import numpy as np
import nibabel as nb
from sklearn.cluster import KMeans
from sklearn.model_selection import GridSearchCV

from nipype.utils.filemanip import fname_presuffix
from nipype.interfaces.base import (
    isdefined,
    traits,
    TraitedSpec as _TraitedSpec,
    BaseInterfaceInputSpec as _BaseInterfaceInputSpec,
    File,
    SimpleInterface,
    OutputMultiObject,
)
from niworkflows.interfaces.bids import ReadSidecarJSON, _ReadSidecarJSONOutputSpec


class _ReadDWIMetadataOutputSpec(_ReadSidecarJSONOutputSpec):
    out_bvec_file = File(desc="corresponding bvec file")
    out_bval_file = File(desc="corresponding bval file")
    out_bmatrix = traits.List(traits.List(traits.Float), desc="b-matrix")


class ReadDWIMetadata(ReadSidecarJSON):
    """
    Extends the NiWorkflows' interface to extract bvec/bval from DWI datasets.
    """

    output_spec = _ReadDWIMetadataOutputSpec

    def _run_interface(self, runtime):
        runtime = super()._run_interface(runtime)

        self._results["out_bvec_file"] = str(self.layout.get_bvec(self.inputs.in_file))
        self._results["out_bval_file"] = str(self.layout.get_bval(self.inputs.in_file))

        bvecs = np.loadtxt(self._results["out_bvec_file"]).T
        bvals = np.loadtxt(self._results["out_bval_file"])

        self._results["out_bmatrix"] = np.hstack((bvecs, bvals[:, np.newaxis])).tolist()

        return runtime


class _WeightedStatInputSpec(_BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True, desc="an image")
    in_weights = traits.List(
        traits.Either(traits.Bool, traits.Float),
        mandatory=True,
        minlen=1,
        desc="list of weights",
    )
    stat = traits.Enum("mean", "std", usedefault=True, desc="statistic to compute")


class _WeightedStatOutputSpec(_TraitedSpec):
    out_file = File(exists=True, desc="masked file")


class WeightedStat(SimpleInterface):
    """Weighted average of the input image across the last dimension."""

    input_spec = _WeightedStatInputSpec
    output_spec = _WeightedStatOutputSpec

    def _run_interface(self, runtime):
        img = nb.load(self.inputs.in_file)
        weights = [float(w) for w in self.inputs.in_weights]
        data = np.asanyarray(img.dataobj)
        statmap = np.average(data, weights=weights, axis=-1)

        self._results["out_file"] = fname_presuffix(
            self.inputs.in_file, suffix=f"_{self.inputs.stat}", newpath=runtime.cwd
        )

        if self.inputs.stat == "std":
            statmap = np.sqrt(
                np.average((data - statmap[..., np.newaxis]) ** 2, weights=weights, axis=-1)
            )

        hdr = img.header.copy()
        img.__class__(
            statmap.astype(hdr.get_data_dtype()),
            img.affine,
            hdr,
        ).to_filename(self._results["out_file"])

        return runtime


class _NumberOfShellsInputSpec(_BaseInterfaceInputSpec):
    in_bvals = File(mandatory=True, desc="bvals file")
    b0_threshold = traits.Float(50, usedefault=True, desc="a threshold for the low-b values")


class _NumberOfShellsOutputSpec(_TraitedSpec):
    models = traits.List(traits.Int, minlen=1, desc="number of shells ordered by model fit")
    n_shells = traits.Int(desc="number of shels")
    out_data = traits.List(
        traits.Float,
        minlen=1,
        desc="new b-values table (after 'shell-fying' DSI)",
    )
    b_values = traits.List(traits.Float, minlen=1, desc="estimated values of b")
    b_masks = traits.List(
        traits.List(traits.Bool, minlen=1),
        minlen=1,
        desc="b-value-wise masks")
    b_indices = traits.List(
        traits.List(traits.Int, minlen=1),
        minlen=1,
        desc="b-value-wise masks")
    b_dict = traits.Dict(
        traits.Int, traits.List(traits.Int), desc="b-values dictionary"
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
        grid_search = GridSearchCV(
            KMeans(), param_grid={"n_clusters": range(1, 10)}, scoring=_rms
        ).fit(in_data[highb_mask].reshape(-1, 1))

        results = np.array(sorted(zip(
            grid_search.cv_results_["mean_test_score"] * -1.0,
            grid_search.cv_results_["param_n_clusters"],
        )))

        self._results["models"] = results[:, 1].astype(int).tolist()
        self._results["n_shells"] = int(grid_search.best_params_["n_clusters"])

        out_data = np.zeros_like(in_data)
        predicted_shell = np.rint(np.squeeze(
            grid_search.best_estimator_.cluster_centers_[
                grid_search.best_estimator_.predict(in_data[highb_mask].reshape(-1, 1))
            ],
        )).astype(int)
        original_bvals = np.unique(np.rint(in_data[highb_mask]).astype(int))

        # If estimated shells matches direct count, probably right -- do not change b-vals
        if len(original_bvals) == self._results["n_shells"]:
            # Find closest b-values
            indices = np.abs(predicted_shell[:, np.newaxis] - original_bvals).argmin(axis=1)
            predicted_shell = original_bvals[indices]

        out_data[highb_mask] = predicted_shell
        self._results["out_data"] = np.round(out_data.astype(float), 2).tolist()
        self._results["b_values"] = sorted(
            np.unique(np.round(predicted_shell.astype(float), 2)).tolist()
        )

        self._results["b_masks"] = [(~highb_mask).tolist()] + [
            np.isclose(self._results["out_data"], bvalue).tolist()
            for bvalue in self._results["b_values"]
        ]
        self._results["b_indices"] = [
            np.atleast_1d(np.squeeze(np.argwhere(b_mask)).astype(int)).tolist()
            for b_mask in self._results["b_masks"]
        ]

        self._results["b_dict"] = {
            int(round(k, 0)): value
            for k, value in zip([0] + self._results["b_values"], self._results["b_indices"])
        }
        return runtime


class _ExtractB0InputSpec(_BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True, desc="dwi file")
    b0_ixs = traits.List(traits.Int, mandatory=True, desc="Index of b0s")


class _ExtractB0OutputSpec(_TraitedSpec):
    out_file = File(exists=True, desc="output b0 file")


class ExtractB0(SimpleInterface):
    """Extract all b=0 volumes from a dwi series."""

    input_spec = _ExtractB0InputSpec
    output_spec = _ExtractB0OutputSpec

    def _run_interface(self, runtime):
        from nipype.utils.filemanip import fname_presuffix

        out_file = fname_presuffix(
            self.inputs.in_file,
            suffix="_b0",
            newpath=runtime.cwd,
        )

        self._results["out_file"] = _extract_b0(
            self.inputs.in_file, self.inputs.b0_ixs, out_path=out_file
        )
        return runtime


class _CorrectSignalDriftInputSpec(_BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True, desc="a 4D file with all low-b volumes")
    bias_file = File(exists=True, desc="a B1 bias field")
    brainmask_file = File(exists=True, desc="a 3D file of the brain mask")
    b0_ixs = traits.List(traits.Int, mandatory=True, desc="Index of b0s")
    bval_file = File(exists=True, mandatory=True, desc="bvalues file")
    full_epi = File(exists=True, desc="a whole DWI dataset to be corrected for drift")


class _CorrectSignalDriftOutputSpec(_TraitedSpec):
    out_file = File(desc="input file after drift correction")
    out_full_file = File(desc="full DWI input after drift correction")
    b0_drift = traits.List(traits.Float)
    signal_drift = traits.List(traits.Float)


class CorrectSignalDrift(SimpleInterface):
    """Correct DWI for signal drift."""

    input_spec = _CorrectSignalDriftInputSpec
    output_spec = _CorrectSignalDriftOutputSpec

    def _run_interface(self, runtime):
        img = nb.load(self.inputs.in_file)
        data = img.get_fdata()
        bmask = np.ones_like(data[..., 0], dtype=bool)

        # Correct for the B1 bias
        if isdefined(self.inputs.bias_file):
            data *= nb.load(self.inputs.bias_file).get_fdata()[..., np.newaxis]

        if isdefined(self.inputs.brainmask_file):
            bmask = np.asanyarray(nb.load(self.inputs.brainmask_file).dataobj) > 1e-3

        global_signal = np.array([
            np.median(data[..., n_b0][bmask]) for n_b0 in range(img.shape[-1])
        ]).astype("float32")

        # Normalize and correct
        global_signal /= global_signal[0]
        self._results["b0_drift"] = [float(gs) for gs in global_signal]

        data *= 1.0 / global_signal[np.newaxis, np.newaxis, np.newaxis, :]

        self._results["out_file"] = fname_presuffix(
            self.inputs.in_file, suffix="_nodrift", newpath=runtime.cwd
        )
        img.__class__(
            data.astype(img.header.get_data_dtype()), img.affine, img.header,
        ).to_filename(self._results["out_file"])

        # Fit line to log-transformed drifts
        K, A_log = np.polyfit(self.inputs.b0_ixs, np.log(global_signal), 1)

        len_dmri = np.loadtxt(self.inputs.bval_file).size
        t_points = np.arange(len_dmri, dtype=int)
        fitted = np.squeeze(_exp_func(t_points, np.exp(A_log), K, 0))
        self._results["signal_drift"] = fitted.astype(float).tolist()

        if isdefined(self.inputs.full_epi):
            self._results["out_full_file"] = fname_presuffix(
                self.inputs.full_epi, suffix="_nodriftfull", newpath=runtime.cwd
            )
            full_img = nb.load(self.inputs.full_epi)
            full_img.__class__(
                (
                    full_img.get_fdata() * fitted[np.newaxis, np.newaxis, np.newaxis, :]
                ).astype(full_img.header.get_data_dtype()),
                full_img.affine,
                full_img.header,
            ).to_filename(self._results["out_full_file"])
        return runtime


class _SplitShellsInputSpec(_BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True, desc="dwi file")
    bvals = traits.List(traits.Float, mandatory=True, desc="bval table")


class _SplitShellsOutputSpec(_TraitedSpec):
    out_file = OutputMultiObject(File(exists=True), desc="output b0 file")


class SplitShells(SimpleInterface):
    """Split a DWI dataset into ."""

    input_spec = _SplitShellsInputSpec
    output_spec = _SplitShellsOutputSpec

    def _run_interface(self, runtime):
        from nipype.utils.filemanip import fname_presuffix

        bval_list = np.rint(self.inputs.bvals).astype(int)
        bvals = np.unique(bval_list)
        img = nb.load(self.inputs.in_file)
        data = np.array(img.dataobj, dtype=img.header.get_data_dtype())

        self._results["out_file"] = []

        for bval in bvals:
            fname = fname_presuffix(
                self.inputs.in_file, suffix=f"_b{bval:05d}", newpath=runtime.cwd
            )
            self._results["out_file"].append(fname)

            img.__class__(
                data[..., np.argwhere(bval_list == bval)],
                img.affine,
                img.header,
            ).to_filename(fname)
        return runtime


class _FilterShellsInputSpec(_BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True, desc="dwi file")
    bvals = traits.List(traits.Float, mandatory=True, desc="bval table")
    bvec_file = File(exists=True, mandatory=True, desc="b-vectors")
    b_threshold = traits.Float(1100, usedefault=True, desc="b-values threshold")


class _FilterShellsOutputSpec(_TraitedSpec):
    out_file = File(exists=True, desc="filtered DWI file")
    out_bvals = traits.List(traits.Float, desc="filtered bvalues")
    out_bvec_file = File(exists=True, desc="filtered bvecs file")
    out_bval_file = File(exists=True, desc="filtered bvals file")


class FilterShells(SimpleInterface):
    """Extract DWIs below a given b-value threshold."""

    input_spec = _FilterShellsInputSpec
    output_spec = _FilterShellsOutputSpec

    def _run_interface(self, runtime):
        from nipype.utils.filemanip import fname_presuffix

        bvals = np.array(self.inputs.bvals)
        bval_mask = bvals < self.inputs.b_threshold
        bvecs = np.loadtxt(self.inputs.bvec_file)[:, bval_mask]

        self._results["out_bvals"] = bvals[bval_mask].astype(float).tolist()
        self._results["out_bvec_file"] = fname_presuffix(
            self.inputs.in_file,
            suffix="_dti.bvec",
            newpath=runtime.cwd,
            use_ext=False,
        )
        np.savetxt(self._results["out_bvec_file"], bvecs)

        self._results["out_bval_file"] = fname_presuffix(
            self.inputs.in_file,
            suffix="_dti.bval",
            newpath=runtime.cwd,
            use_ext=False,
        )
        np.savetxt(self._results["out_bval_file"], bvals)

        self._results["out_file"] = fname_presuffix(
            self.inputs.in_file,
            suffix="_dti",
            newpath=runtime.cwd,
        )

        dwi_img = nb.load(self.inputs.in_file)
        data = np.array(dwi_img.dataobj, dtype=dwi_img.header.get_data_dtype())[..., bval_mask]
        dwi_img.__class__(
            data,
            dwi_img.affine,
            dwi_img.header,
        ).to_filename(self._results["out_file"])

        return runtime


class _DipyDTIInputSpec(_BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True, desc="dwi file")
    bvals = traits.List(traits.Float, mandatory=True, desc="bval table")
    bvec_file = File(exists=True, mandatory=True, desc="b-vectors")
    brainmask = File(exists=True, desc="brain mask file")
    free_water_model = traits.Bool(False, usedefault=True, desc="use free water model")
    b_threshold = traits.Float(1100, usedefault=True, desc="use only inner shells of the data")


class _DipyDTIOutputSpec(_TraitedSpec):
    out_fa = File(exists=True, desc="output FA file")
    out_md = File(exists=True, desc="output MD file")


class DipyDTI(SimpleInterface):
    """Split a DWI dataset into ."""

    input_spec = _DipyDTIInputSpec
    output_spec = _DipyDTIOutputSpec

    def _run_interface(self, runtime):
        from dipy.core.gradients import gradient_table_from_bvals_bvecs
        from dipy.reconst.dti import TensorModel
        from dipy.reconst.fwdti import FreeWaterTensorModel
        from nipype.utils.filemanip import fname_presuffix

        bvals = np.array(self.inputs.bvals)
        bval_mask = bvals < self.inputs.b_threshold

        gtab = gradient_table_from_bvals_bvecs(
            bvals=bvals[bval_mask],
            bvecs=np.loadtxt(self.inputs.bvec_file).T[bval_mask],
        )

        img = nb.load(self.inputs.in_file)
        data = img.get_fdata(dtype="float32")[..., bval_mask]

        brainmask = np.ones_like(data[..., 0], dtype=bool)

        if isdefined(self.inputs.brainmask):
            brainmask = np.asanyarray(nb.load(self.inputs.brainmask).dataobj) > 0.5

        DTIModel = FreeWaterTensorModel if self.inputs.free_water_model else TensorModel

        # Fit DIT
        fwdtifit = DTIModel(gtab).fit(
            data,
            mask=brainmask,
        )

        # Extract the FA
        fa_data = np.array(fwdtifit.fa, dtype="float32")
        fa_data[np.isnan(fa_data)] = 0
        fa_data = np.clip(fa_data, 0, 1)

        fa_nii = nb.Nifti1Image(
            fa_data,
            img.affine,
            None,
        )

        fa_nii.header.set_xyzt_units("mm")
        fa_nii.header.set_intent("estimate", name="Fractional Anisotropy (FA)")
        fa_nii.header["cal_max"] = 1.0
        fa_nii.header["cal_min"] = 0.0

        self._results["out_fa"] = fname_presuffix(
            self.inputs.in_file,
            suffix="fa",
            newpath=runtime.cwd,
        )

        fa_nii.to_filename(self._results["out_fa"])

        # Extract the AD
        self._results["out_md"] = fname_presuffix(
            self.inputs.in_file,
            suffix="md",
            newpath=runtime.cwd,
        )
        ad_data = np.array(fwdtifit.ad, dtype="float32")
        ad_data[np.isnan(ad_data)] = 0
        ad_data = np.clip(ad_data, 0, 1)
        ad_hdr = fa_nii.header.copy()
        ad_hdr.set_intent("estimate", name="Mean diffusivity (MD)")
        nb.Nifti1Image(
            ad_data,
            img.affine,
            ad_hdr
        ).to_filename(self._results["out_md"])

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


def _extract_b0(in_file, b0_ixs, out_path=None):
    """Extract the *b0* volumes from a DWI dataset."""
    if out_path is None:
        out_path = fname_presuffix(in_file, suffix="_b0")

    img = nb.load(in_file)
    bzeros = np.squeeze(np.asanyarray(img.dataobj)[..., b0_ixs])

    hdr = img.header.copy()
    hdr.set_data_shape(bzeros.shape)
    hdr.set_xyzt_units("mm")
    nb.Nifti1Image(bzeros, img.affine, hdr).to_filename(out_path)
    return out_path


def _exp_func(t, A, K, C):
    return A * np.exp(K * t) + C
