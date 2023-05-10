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
    traits,
    TraitedSpec as _TraitedSpec,
    BaseInterfaceInputSpec as _BaseInterfaceInputSpec,
    File,
    SimpleInterface,
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


class _WeightedAverageInputSpec(_BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True, desc="an image")
    in_weights = traits.List(
        traits.Either(traits.Bool, traits.Float),
        mandatory=True,
        minlen=1,
        desc="list of weights",
    )


class _WeightedAverageOutputSpec(_TraitedSpec):
    out_file = File(exists=True, desc="masked file")


class WeightedAverage(SimpleInterface):
    """Weighted average of the input image across the last dimension."""

    input_spec = _WeightedAverageInputSpec
    output_spec = _WeightedAverageOutputSpec

    def _run_interface(self, runtime):
        img = nb.load(self.inputs.in_file)
        weights = [float(w) for w in self.inputs.in_weights]
        average = np.average(np.asanyarray(img.dataobj), weights=weights, axis=-1)

        self._results["out_file"] = fname_presuffix(
            self.inputs.in_file, suffix="_average", newpath=runtime.cwd
        )

        hdr = img.header.copy()
        img.__class__(
            average.astype(hdr.get_data_dtype()),
            img.affine,
            hdr,
        ).to_filename(self._results["out_file"])

        return runtime


class _NumberOfShellsInputSpec(_BaseInterfaceInputSpec):
    in_data = traits.List(
        traits.Float,
        mandatory=True,
        minlen=1,
        desc="list of weights",
    )
    b0_threshold = traits.Float(50, usedefault=True, desc="a threshold for the low-b values")


class _NumberOfShellsOutputSpec(_TraitedSpec):
    models = traits.List(traits.Int, minlen=1, desc="number of shells ordered by model fit")
    n_shells = traits.Int(desc="number of shels")
    b_values = traits.List(traits.Float, minlen=1, desc="estimated values of b")
    out_data = traits.List(traits.Float, minlen=1, desc="new b-values")
    lowb_mask = traits.List(traits.Bool)
    lowb_indices = traits.List(traits.Int)


class NumberOfShells(SimpleInterface):
    """
    Weighted average of the input image across the last dimension.

    Examples
    --------
    >>> NumberOfShells(in_data=[0] * 8 + [1000] * 12 + [2000] * 10).run().outputs.n_shells
    2
    >>> NumberOfShells(in_data=[0] * 8 + [1000] * 12).run().outputs.n_shells
    1
    >>> NumberOfShells(in_data=np.arange(0, 9000, 120).tolist()).run().outputs.n_shells
    9

    """

    input_spec = _NumberOfShellsInputSpec
    output_spec = _NumberOfShellsOutputSpec

    def _run_interface(self, runtime):
        in_data = np.array(self.inputs.in_data)
        highb_mask = in_data > self.inputs.b0_threshold
        grid_search = GridSearchCV(
            KMeans(), param_grid={"n_clusters": range(1, 10)}, scoring=_rms
        ).fit(in_data[highb_mask].reshape(-1, 1))

        results = sorted(zip(
            grid_search.cv_results_["mean_test_score"] * -1.0,
            grid_search.cv_results_["param_n_clusters"],
        ))

        self._results["lowb_mask"] = (~highb_mask).tolist()
        self._results["lowb_indices"] = np.squeeze(np.argwhere(~highb_mask)).tolist()
        self._results["models"] = list(list(zip(*results))[1])
        self._results["n_shells"] = grid_search.best_params_["n_clusters"]

        self._results["b_values"] = sorted(
            np.squeeze(grid_search.best_estimator_.cluster_centers_).tolist()
        )

        out_data = np.zeros_like(in_data)
        predicted_shell = grid_search.best_estimator_.cluster_centers_[
            grid_search.best_estimator_.predict(in_data[highb_mask].reshape(-1, 1))
        ]
        out_data[highb_mask] = np.squeeze(predicted_shell)
        self._results["out_data"] = out_data.tolist()
        return runtime


def _rms(estimator, X):
    """
    Callable to pass to GridSearchCV that will calculate a distance score.

    To consider: using `MDL
    <http://erikerlandson.github.io/blog/2016/08/03/x-medoids-using-minimum-description-length-to-identify-the-k-in-k-medoids/>`__

    """
    if len(np.unique(estimator.cluster_centers_)) < estimator.n_clusters:
        return -np.inf

    # Calculate distance from assigned shell centroid
    distance = X - estimator.cluster_centers_[estimator.predict(X)]
    # Make negative so CV optimizes minimizes the error
    return -np.sqrt(distance**2).sum()
