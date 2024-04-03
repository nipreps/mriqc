
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
    of Noise) algorithm [Koay2009]_ to estimate the standard deviation (sigma) of the
    noise in each voxel of a 4D dMRI data array.

.. _iqms_cc_snr:

SNR estimated in the Corpus Callosum (``cc_snr``)
    Worst-case and best-case signal-to-noise ratio (SNR) within the corpus callosum.

.. _iqms_fa_nans:

Number of not-a-number (NaN) values in the FA map (``fa_nan``)
    Fraction of NaNs within the brain mask, in ppm.

.. _iqms_fa_degenerate:

Number of degenerate modeled voxels in the FA map (``fa_degenerate``)
    Fraction of invalid FA values (i.e., outside the [0, 1] closed range) within
    the brain mask, in ppm.

IQMs relating artifacts and other
---------------------------------
IQMs targeting artifacts that are specific of DWI images.

.. _iqms_spike_ppm:

Global and slice-wise spike fractions (``spikes_ppm``)
    Fractions of voxels classified as spikes (in parts-per-million, ppm).
    The spikes mask is calculated by identifying voxels with signal intensities
    exceeding a threshold based on standard deviations above the mean.

References
----------
.. [Koay2009] Koay C.G., E. Ozarslan, C. Pierpaoli. Probabilistic Identification
       and Estimation of Noise (PIESNO): A self-consistent approach and
       its applications in MRI. JMR, 199(1):94-103, 2009.

"""
from __future__ import annotations

import numpy as np
from statsmodels.robust.scale import mad


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

    Parameters
    ----------
    in_b0 : :obj:`~numpy.ndarray`
        The 3D or 4D dMRI data array. If 4D, the first volume (assumed to be
        the $b$=0 image) is used for noise estimation.
    percentiles : :obj:`tuple` (float, float, float), optional (default=(25, 50, 75))
        A tuple of three integers specifying the percentiles of the variance
        distribution to use for noise estimation. These percentiles represent
        different noise levels within the data.
    mask : :obj:`~numpy.ndarray`, optional (default=``None``)
        A boolean mask used to restrict the noise estimation to specific brain regions.
        If ``None``, a mask of ones with the same shape as the first 3 dimensions of
        ``in_b0`` is used.

    Returns
    -------
    :obj:`dict`
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
    std_signal = mad(in_b0[cc_mask])

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

    return cc_snr_estimates, std_signal


def spike_ppm(
    spike_mask: np.ndarray,
    slice_threshold: float = 0.05,
    decimals: int = 8,
) -> dict[str, float | np.ndarray]:
    """
    Calculates fractions (global and slice-wise) of voxels classified as spikes in ppm.

    This function computes two metrics:

    * Global spike parts-per-million [ppm]: Fraction of voxels exceeding the spike
      threshold across the entire data array.
    * Slice-wise spiking [ppm]: The fraction of slices along each dimension of
      the data array where the average fraction of spiking voxels within the slice
      exceeds a user-defined threshold (``slice_threshold``).

    Parameters
    ----------
    spike_mask : :obj:`~numpy.ndarray` (bool, same shape as data)
        The binary mask indicating spike voxels (True) and non-spike voxels (False).
    slice_threshold : :obj:`float`, optional (default=0.05)
        The minimum fraction of voxels in a slice that must be classified as spikes
        for the slice to be considered spiking.
    decimals : :obj:`int`
        The number of decimals to round the fractions.

    Returns
    -------
    :obj:`dict`
        A dictionary containing the calculated spike percentages:

        * 'global': :obj:`float` - global spiking voxels ppm.
        * 'slice': :obj:`list` of :obj:`float` - List of slice-wise spiking voxel
          fractions in ppm for each dimension of the data array.

    """

    spike_perc_global = round(float(np.mean(np.ravel(spike_mask))), decimals) * 1e6
    spike_perc_slice = [
        round(
            float(np.mean(np.mean(spike_mask, axis=axis) > slice_threshold)), decimals
        ) * 1e6
        for axis in range(spike_mask.ndim)
    ]

    return {'global': spike_perc_global, 'slice': spike_perc_slice}


def neighboring_dwi_correlation(
    dwi_data: np.ndarray,
    neighbor_indices: list[tuple[int, int]],
    mask: np.ndarray | None = None,
) -> float:
    """
    Calculates the Neighboring DWI Correlation (NDC) from diffusion MRI (dMRI) data.

    The NDC is a measure of the correlation between signal intensities in neighboring
    diffusion-weighted images (DWIs) within a mask. A low NDC (typically below 0.4)
    can indicate poor image quality, according to Yeh et al. [Yeh2019]_.

    Parameters
    ----------
    dwi_data : 4D :obj:`~numpy.ndarray`
        DWI data on which to calculate NDC
    neighbor_indices : :obj:`list` of :obj:`tuple`
        List of (from, to) index neighbors.
    mask : 3D :obj:`~numpy.ndarray`, optional
        optional mask of voxels to include in the NDC calculation

    Returns
    -------
    :obj:`float`
        The NDC value.

    References
    ----------
    .. [Yeh2019] Yeh, Fang-Cheng, et al. "Differential tractography as a
                 track-based biomarker for neuronal injury."
                 NeuroImage 202 (2019): 116131.

    Notes
    -----
    This is a copy of DIPY's code to be removed (and just imported) as soon as
    a new release of DIPY is cut including
    `dipy/dipy#3156 <https://github.com/dipy/dipy/pull/3156>`__.

    """

    neighbor_correlations = []

    mask = np.ones_like(dwi_data[..., 0], dtype=bool) if mask is None else mask

    dwi_data = dwi_data[mask]

    for from_index, to_index in neighbor_indices:
        flat_from_image = dwi_data[from_index]
        flat_to_image = dwi_data[to_index]

        neighbor_correlations.append(
            np.corrcoef(flat_from_image, flat_to_image)[0, 1]
        )

    return np.round(np.mean(neighbor_correlations), 4)
