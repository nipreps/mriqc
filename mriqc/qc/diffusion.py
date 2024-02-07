
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
from dipy.segment.mask import segment_from_cfa
from dipy.segment.mask import bounding_box

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


def cc_snr(data, gtab, bmag=None, mask=None):
    """
    Calculate worse-/best-case signal-to-noise ratio in the corpus callosum

    Parameters
    ----------
    data : ndarray

    gtab : GradientTable class instance or tuple

    bmag : int
        From dipy.core.gradients:
        The order of magnitude that the bvalues have to differ to be
        considered an unique b-value. B-values are also rounded up to
        this order of magnitude. Default: derive this value from the
        maximal b-value provided: $bmag=log_{10}(max(bvals)) - 1$.

    mask : numpy array
        Boolean brain mask


    """
    if isinstance(gtab, GradientTable):
        pass

    if mask is None:
        mask = np.ones(data.shape[:3])

    tenmodel = TensorModel(gtab)
    tensorfit = tenmodel.fit(data, mask=mask)

    threshold = (0.6, 1, 0, 0.1, 0, 0.1)
    CC_box = np.zeros_like(data[..., 0])

    mins, maxs = bounding_box(mask) #mask needs to be volume
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

    b0_data = data[..., gtab.b0s_mask]
    std_signal = np.std(b0_data[mask_cc_part], axis=-1)

    # Per-shell calculation
    rounded_bvals = round_bvals(gtab.bvals, bmag)
    bvals = unique_bvals_magnitude(gtab.bvals, bmag)

    cc_snr_best = np.zeros(gtab.bvals.shape)
    cc_snr_worst = np.zeros(gtab.bvals.shape)

    for ind, bval in enumerate(bvals):
        if bval == 0:
            mean_signal = np.mean(data[..., rounded_bvals == 0], axis=-1)
            cc_snr_worst[ind] = np.mean(mean_signal/std_signal)
            cc_snr_best[ind] = np.mean(mean_signal/std_signal)
            continue

        bval_data = data[..., rounded_bvals == bval]
        bval_bvecs = gtab.bvecs[rounded_bvals == bval]

        axis_X = np.argmin(np.sum((bval_bvecs-np.array([1, 0, 0]))**2, axis=-1))
        axis_Y = np.argmin(np.sum((bval_bvecs-np.array([0, 1, 0]))**2, axis=-1))
        axis_Z = np.argmin(np.sum((bval_bvecs-np.array([0, 0, 1]))**2, axis=-1))

        data_X = bval_data[..., axis_X]
        data_Y = bval_data[..., axis_Y]
        data_Z = bval_data[..., axis_Z]

        mean_signal_X = np.mean(data_X[mask_cc_part])
        mean_signal_Y = np.mean(data_Y[mask_cc_part])
        mean_signal_Z = np.mean(data_Z[mask_cc_part])

        cc_snr_worst[ind] = np.mean(mean_signal_X/std_signal)
        cc_snr_best[ind] = np.mean(np.mean(mean_signal_Y, mean_signal_Z)/std_signal)

    return cc_snr_worst, cc_snr_best


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