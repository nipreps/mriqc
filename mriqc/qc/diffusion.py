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

import numpy as np


"""
Image quality metrics for diffusion MRI data
============================================
"""

def noise_func(img, gtab):
    pass

def get_spike_mask(data, grouping_vals, z_threshold=3):
    """
    Return binary mask of spike/no spike

    Parameters
    ----------
    data : numpy array
        Data to be thresholded
    grouping_vals : numpy array
        Values by which to group data for thresholding (bvals or full mask)
    z_threshold : :obj:`float`
        Number of standard deviations above the mean to use as spike threshold

    Returns
    ---------
    numpy array
    """

    threshold_mask = np.zeros(data.shape)
    gvals = np.unique(grouping_vals) # Not the right function
    gval_inds = np.where(grouping_vals == gval) # Not the right functions

    if grouping_vals.shape == data.shape:
        for gval in gvals:
            gval_data = data[gval_inds]
            gval_threshold = (z_threshold*np.std(gval_data)) + np.mean(gval_data)
            threshold_mask[gval_inds] = gval_threshold*np.ones(gval_data.shape)
    else:
        for gval in gvals:
            gval_data = data[...,gval_inds]
            gval_threshold = (z_threshold*np.std(gval_data)) + np.mean(gval_data)
            threshold_mask[...,gval_inds] = gval_threshold*np.ones(gval_data.shape)

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