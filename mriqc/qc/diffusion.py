
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

IQMs relating to spatial information
------------------------------------
Definitions are given in the :ref:`summary of structural IQMs <iqms_t1w>`.

.. _iqms_efc:

- **Entropy-focus criterion** (:py:func:`~mriqc.qc.anatomical.efc`).

.. _iqms_fber:

- **Foreground-Background energy ratio** (:py:func:`~mriqc.qc.anatomical.fber`,  [Shehzad2015]_).

.. _iqms_fwhm:

- **Full-width half maximum smoothness** (``fwhm_*``, see [Friedman2008]_).

.. _iqms_snr:

- **Signal-to-noise ratio** (:py:func:`~mriqc.qc.anatomical.snr`).

.. _iqms_summary:

- **Summary statistics** (:py:func:`~mriqc.qc.anatomical.summary_stats`).

IQMs relating to diffusion weighting
------------------------------------
IQMs specific to diffusion weighted imaging.

.. _iqms_piesno:

Noise in raw dMRI estimated with PIESNO (``piesno_sigma``)
    Employs PIESNO (Probabilistic Identification and Estimation
    of Noise) algorithm [1]_ to estimate the standard deviation (sigma) of the
    noise in each voxel of a 4D dMRI data array.

.. _iqms_cc_snr:

SNR estimated in the Corpus Callosum (``cc_snr``)
    Worst-case and best-case signal-to-noise ratio (SNR) within the corpus callosum.

IQMs relating artifacts and other
---------------------------------
IQMs targeting artifacts that are specific of DWI images.

.. _iqms_spike_percentage:

Global and slice-wise spike percentages (``spike_perc``)
    Voxels classified as spikes. The spikes mask is calculated by identifying
    voxels with signal intensities exceeding a threshold based on standard
    deviations above the mean.

"""
from __future__ import annotations

import numpy as np


def noise_b0(
    in_b0: np.ndarray,
    percentiles: tuple[float, float, float] = (25., 50., 75.),
    mask: np.ndarray | None = None
) -> dict[str, float]:
    """
    Estimates noise levels in raw dMRI data using the variance of the $b$=0 volumes.

    This function calculates the variance of the $b$=0 volumes in a 4D dMRI array
    within a provided mask. It then computes the noise estimates at specified
    percentiles of the variance distribution. This approach assumes that noise primarily
    contributes to the lower end of the variance distribution.

    Parameters:
    ----------
    in_b0 : :obj:`~numpy.ndarray`
        The 3D or 4D dMRI data array. If 4D, the first volume (assumed to be
        the $b$=0 image) is used for noise estimation.
    percentiles : :obj:`tuple`(float, float, float), optional (default=(25, 50, 75))
        A tuple of three integers specifying the percentiles of the variance
        distribution to use for noise estimation. These percentiles represent
        different noise levels within the data.
    mask : :obj:`~numpy.ndarray`, optional (default=``None``)
        A boolean mask used to restrict the noise estimation to specific brain regions.
        If ``None``, a mask of ones with the same shape as the first 3 dimensions of
        ``in_b0`` is used.

    Returns:
    -------
    noise_estimates : :obj:`dict`
        A dictionary containing the noise estimates at the specified percentiles:
            * keys: :obj:`str` - Percentile values (e.g., '25', '50', '75').
            * values: :obj:`float` - Noise level estimates at the corresponding percentiles.
    """

    if in_b0.ndim != 4:
        return None

    data = in_b0[
        np.ones(in_b0.shape[:3], dtype=bool) if mask is None else mask
    ]
    variance = np.var(data, -1)
    noise_estimates = dict(zip(
        (f'{p}' for p in percentiles),
        np.percentile(variance, percentiles),
    ))

    return noise_estimates


def noise_piesno(data: np.ndarray, n_channels: int = 4) -> (np.ndarray, np.ndarray):
    """
    Estimates noise in raw diffusion MRI (dMRI) data using the PIESNO algorithm.

    This function implements the PIESNO (Probabilistic Identification and Estimation
    of Noise) algorithm [1]_ to estimate the standard deviation (sigma) of the
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

    Notes
    -----

    .. [1] Koay C.G., E. Ozarslan, C. Pierpaoli. Probabilistic Identification
           and Estimation of Noise (PIESNO): A self-consistent approach and
           its applications in MRI. JMR, 199(1):94-103, 2009.
    """
    from dipy.denoise.noise_estimate import piesno

    sigma, mask = piesno(data, N=n_channels, return_mask=True)
    return sigma, mask


def cc_snr(
    in_b0: np.ndarray,
    dwi_shells: list[np.ndarray],
    cc_mask: np.ndarray,
    b_values: np.ndarray,
    b_vectors: np.ndarray,
) -> dict[int, (float, float)]:
    """
    Calculates the worst-case and best-case signal-to-noise ratio (SNR) within the corpus callosum.

    This function estimates the SNR in the corpus callosum (CC) by comparing the
    mean signal intensity within the CC mask to the standard deviation of the background
    signal (extracted from the b0 image). It performs separate calculations for
    each diffusion-weighted imaging (DWI) shell.

    **Worst-case SNR:** The mean signal intensity along the diffusion direction with the
    lowest signal is considered the worst-case scenario.

    **Best-case SNR:** The mean signal intensity averaged across the two diffusion
    directions with the highest signal is considered the best-case scenario.

    Parameters
    ----------
    in_b0 : :obj:`~numpy.ndarray` (float, 3D)
        T1-weighted or b0 image used for background signal estimation.
    dwi_shells : list[:obj:`~numpy.ndarray` (float, 4D)]
        List of DWI data for each diffusion shell.
    cc_mask : :obj:`~numpy.ndarray` (bool, 3D)
        Boolean mask of the corpus callosum.
    b_values : :obj:`~numpy.ndarray` (int)
        Array of b-values for each DWI volume in ``dwi_shells``.
    b_vectors : :obj:`~numpy.ndarray` (float)
        Array of diffusion-encoding vectors for each DWI volume in ``dwi_shells``.

    Returns
    -------
    cc_snr_estimates : :obj:`dict`
        Dictionary containing SNR estimates for each b-value. Keys are the b-values
        (integers), and values are tuples containing two elements:
            * The first element is the worst-case SNR (float).
            * The second element is the best-case SNR (float).
    """

    cc_mask = cc_mask > 0  # Ensure it's a boolean mask
    std_signal = in_b0[cc_mask].std()

    cc_snr_estimates = {}

    xyz = np.eye(3)

    b_values = np.rint(b_values).astype(np.uint16)

    # Shell-wise calculation
    for bval, bvecs, shell_data in zip(b_values, b_vectors, dwi_shells):
        if bval == 0:
            cc_snr_estimates[f'b{bval:d}'] = in_b0[cc_mask].mean() / std_signal
            continue

        shell_data = shell_data[cc_mask]

        # Find main directions of diffusion
        axis_X = np.argmin(np.sum(
            (bvecs - xyz[0, :]) ** 2, axis=-1))
        axis_Y = np.argmin(np.sum(
            (bvecs - xyz[1, :]) ** 2, axis=-1))
        axis_Z = np.argmin(np.sum(
            (bvecs - xyz[2, :]) ** 2, axis=-1))

        data_X = shell_data[..., axis_X]
        data_Y = shell_data[..., axis_Y]
        data_Z = shell_data[..., axis_Z]

        mean_signal_worst = np.mean(data_X)
        mean_signal_best = 0.5 * (np.mean(data_Y) + np.mean(data_Z))

        cc_snr_estimates[f'b{bval:d}'] = (
            np.mean(mean_signal_worst / std_signal),
            np.mean(mean_signal_best / std_signal),
        )

    return cc_snr_estimates


def spike_percentage(
    data: np.ndarray,
    spike_mask: np.ndarray,
    slice_threshold: float = 0.05,
) -> dict[str, float | np.ndarray]:
    """
    Calculates the percentage of voxels classified as spikes (global and slice-wise).

    This function computes two metrics:

    * Global spike percentage: The average fraction of voxels exceeding the spike
      threshold across the entire data array.
    * Slice-wise spiking percentage: The fraction of slices along each dimension of
      the data array where the average fraction of spiking voxels within the slice
      exceeds a user-defined threshold (``slice_threshold``).

    Parameters:
    ----------
    data : :obj:`~numpy.ndarray` (float, 4D)
        The data array used to generate the spike mask.
    spike_mask : :obj:`~numpy.ndarray` (bool, same shape as data)
        The binary mask indicating spike voxels (True) and non-spike voxels (False).
    slice_threshold : :obj:`float`, optional (default=0.05)
        The minimum fraction of voxels in a slice that must be classified as spikes
        for the slice to be considered spiking.

    Returns:
    -------
    :obj:`dict`
        A dictionary containing the calculated spike percentages:
            * 'spike_perc_global': :obj:`float` - Global percentage of spiking voxels.
            * 'spike_perc_slice': :obj:`list` of :obj:`float` - List of slice-wise
              spiking percentages for each dimension of the data array.
    """

    spike_perc_global = float(np.mean(np.ravel(spike_mask)))
    spike_perc_slice = [
        float(np.mean(np.mean(spike_mask, axis=axis) > slice_threshold))
        for axis in range(data.ndim)
    ]

    return {'spike_perc_global': spike_perc_global, 'spike_perc_slice': spike_perc_slice}
