
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

"""
Image quality metrics for diffusion MRI data
============================================
"""

import numpy as np
from dipy.core.gradients import gradient_table
from dipy.core.gradients import GradientTable
from dipy.reconst.dti import TensorModel
from dipy.denoise.noise_estimate import piesno
from dipy.core.gradients import unique_bvals_magnitude
from dipy.core.gradients import round_bvals

def noise_func(img, gtab):
    pass

def noise_b0(data, gtab, mask=None):
    """
    Estimate noise in raw dMRI based on b0 variance.

    Parameters
    ----------
    """
    if mask is None:
        mask = np.ones(data.shape[:3], dtype=bool)
    b0 = data[..., ~gtab.b0s_mask]
    return np.percentile(np.var(b0[mask], -1), (25, 50, 75))


def noise_piesno(data, n_channels=4):
    """
    Estimate noise in raw dMRI data using the PIESNO [1]_ algorithm.


    Parameters
    ----------

    Returns
    -------


    Notes
    -----

    .. [1] Koay C.G., E. Ozarslan, C. Pierpaoli. Probabilistic Identification
           and Estimation of Noise (PIESNO): A self-consistent approach and
           its applications in MRI. JMR, 199(1):94-103, 2009.
    """
    sigma, mask = piesno(data, N=n_channels, return_mask=True)
    return sigma, mask


def cc_snr(data, gtab):
    """
    Calculate worse-/best-case signal-to-noise ratio in the corpus callosum

    Parameters
    ----------
    data : ndarray

    gtab : GradientTable class instance or tuple

    """
    if isinstance(gtab, GradientTable):
        pass

    # XXX Per-shell calculation
    tenmodel = TensorModel(gtab)
    tensorfit = tenmodel.fit(data, mask=mask)

    from dipy.segment.mask import segment_from_cfa
    from dipy.segment.mask import bounding_box

    threshold = (0.6, 1, 0, 0.1, 0, 0.1)
    CC_box = np.zeros_like(data[..., 0])

    mins, maxs = bounding_box(mask)
    mins = np.array(mins)
    maxs = np.array(maxs)
    diff = (maxs - mins) // 4
    bounds_min = mins + diff
    bounds_max = maxs - diff

    CC_box[bounds_min[0]:bounds_max[0],
        bounds_min[1]:bounds_max[1],
        bounds_min[2]:bounds_max[2]] = 1

    mask_cc_part, cfa = segment_from_cfa(tensorfit, CC_box, threshold,
                                        return_cfa=True)

    mean_signal = np.mean(data[mask_cc_part], axis=0)


def get_spike_mask(data, z_threshold=3, grouping_vals=None, bmag=None):
    """
    Return binary mask of spike/no spike

    Parameters
    ----------
    data : numpy array
        Data to be thresholded
    z_threshold : :obj:`float`
        Number of standard deviations above the mean to use as spike threshold
    grouping_vals : numpy array
        Values by which to group data for thresholding (bvals or full mask)
    bmag : int
        From dipy.core.gradients: 
        The order of magnitude that the bvalues have to differ to be
        considered an unique b-value. B-values are also rounded up to
        this order of magnitude. Default: derive this value from the
        maximal b-value provided: $bmag=log_{10}(max(bvals)) - 1$.

    Returns
    ---------
    numpy array
    """

    if grouping_vals is None:
        threshold = (z_threshold*np.std(data)) + np.mean(data)
        spike_mask = data > threshold
        return spike_mask

    threshold_mask = np.zeros(data.shape)

    rounded_grouping_vals = round_bvals(grouping_vals, bmag)
    gvals = unique_bvals_magnitude(grouping_vals, bmag)

    if grouping_vals.shape == data.shape:
        for gval in gvals:
            gval_data = data[rounded_grouping_vals == gval]
            gval_threshold = (z_threshold*np.std(gval_data)) + np.mean(gval_data)
            threshold_mask[rounded_grouping_vals == gval] = gval_threshold*np.ones(gval_data.shape)
    else:
        for gval in gvals:
            gval_data = data[..., rounded_grouping_vals == gval]
            gval_threshold = (z_threshold*np.std(gval_data)) + np.mean(gval_data)
            threshold_mask[..., rounded_grouping_vals == gval] = gval_threshold*np.ones(gval_data.shape)

    spike_mask = data > threshold_mask

    return spike_mask


def get_slice_spike_percentage(data, z_threshold=3, slice_threshold=.05):
    """
    Return percentage of slices spiking along each dimension

    Parameters
    ----------
    data : numpy array
        Data to be thresholded
    z_threshold : :obj:`float`
        Number of standard deviations above the mean to use as spike threshold
    slice_threshold : :obj:`float`
        Percentage of slice elements that need to be above spike threshold for slice to be considered spiking

    Returns
    ---------
    array
    """
    spike_mask = get_spike_mask(data, z_threshold)

    ndim = data.ndim
    slice_spike_percentage = np.zeros(ndim)

    for ii in range(ndim):
        slice_spike_percentage[ii] = np.mean(np.mean(spike_mask, ii) > slice_threshold)

    return slice_spike_percentage


def get_global_spike_percentage(data, z_threshold=3):
    """
    Return percentage of array elements spiking

    Parameters
    ----------
    data : numpy array
        Data to be thresholded
    z_threshold : :obj:`float`
        Number of standard deviations above the mean to use as spike threshold

    Returns
    ---------
    float
    """
    spike_mask = get_spike_mask(data, z_threshold)
    global_spike_percentage = np.mean(np.ravel(spike_mask))

    return global_spike_percentage

def noise_func_for_shelled_data(shelled_data, gtab):
    pass