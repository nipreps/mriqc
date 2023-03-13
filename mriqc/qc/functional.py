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
r"""
Measures for the spatial information
====================================

Definitions are given in the
:ref:`summary of structural IQMs <iqms_t1w>`.

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


Measures for the temporal information
-------------------------------------

.. _iqms_dvars :

DVARS
  D referring to temporal derivative of timecourses, VARS referring to
  RMS variance over voxels ([Power2012]_ ``dvars_nstd``) indexes the rate of change of
  BOLD signal across the entire brain at each frame of data. DVARS is calculated
  `with nipype
  <https://nipype.readthedocs.io/en/latest/api/generated/nipype.algorithms.confounds.html#computedvars>`_
  after motion correction:

  .. math ::

      \text{DVARS}_t = \sqrt{\frac{1}{N}\sum_i \left[x_{i,t} - x_{i,t-1}\right]^2}


  .. note ::

    Intensities are scaled to 1000 leading to the units being expressed in x10
    :math:`\%\Delta\text{BOLD}` change.

  .. note ::

    MRIQC calculates two additional standardized values of the DVARS.
    The ``dvars_std`` metric is normalized with the standard deviation of the
    temporal difference time series. The ``dvars_vstd`` is a voxel-wise
    standardization of DVARS, where the temporal difference time series is
    normalized across time by that voxel standard deviation across time, before
    computing the RMS of the temporal difference [Nichols2013]_.

.. _iqms_gcor:

Global Correlation (``gcor``)
  calculates an optimized summary of time-series
  correlation as in [Saad2013]_ using AFNI's ``@compute_gcor``:

  .. math ::

      \text{GCOR} = \frac{1}{N}\mathbf{g}_u^T\mathbf{g}_u

  where :math:`\mathbf{g}_u` is the average of all unit-variance time series in a
  :math:`T` (# timepoints) :math:`\times` :math:`N` (# voxels) matrix.

.. _iqms_tsnr:

Temporal SNR (:abbr:`tSNR (temporal SNR)`, ``tsnr``)
  is a simplified interpretation of the tSNR definition [Kruger2001]_.
  We report the median value
  of the `tSNR map
  <https://nipype.readthedocs.io/en/latest/api/generated/nipype.algorithms.confounds.html#tsnr>`_
  calculated like:

  .. math ::

      \text{tSNR} = \frac{\langle S \rangle_t}{\sigma_t},

  where :math:`\langle S \rangle_t` is the average BOLD signal (across time),
  and :math:`\sigma_t` is the corresponding temporal standard-deviation map. Higher
  values are better.


Measures for artifacts and other
--------------------------------

.. _iqms_fd:

Framewise Displacement
  expresses instantaneous head-motion [Jenkinson2002]_.
  MRIQC reports the average FD, labeled as ``fd_mean``.
  Rotational displacements are calculated as the displacement on the surface of a
  sphere of radius 50 mm [Power2012]_:

  .. math ::

      \text{FD}_t = |\Delta d_{x,t}| + |\Delta d_{y,t}| +
      |\Delta d_{z,t}| + |\Delta \alpha_t| + |\Delta \beta_t| + |\Delta \gamma_t|

  Along with the base framewise displacement, MRIQC reports the
  **number of timepoints above FD threshold** (``fd_num``), and the
  **percent of FDs above the FD threshold** w.r.t. the full timeseries (``fd_perc``).
  In both cases, the threshold is set at 0.20mm.

.. _iqms_gsr:

Ghost to Signal Ratio (:py:func:`~mriqc.qc.functional.gsr`)
  labeled in the reports as ``gsr_x`` and ``gsr_y``
  (calculated along the two possible phase-encoding axes **x**, **y**):

  .. math ::

      \text{GSR} = \frac{\mu_G - \mu_{NG}}{\mu_S}

  .. image :: ../_static/epi-gsrmask.png
    :width: 200px
    :align: center

.. _iqms_aor:

AFNI's outlier ratio (``aor``)
  Mean fraction of outliers per fMRI volume
  as given by AFNI's ``3dToutcount``.

.. _iqms_aqi:

AFNI's quality index (``aqi``)
  Mean quality index as computed by AFNI's ``3dTqual``; for each volume,
  it is one minus the Spearman's (rank) correlation of that volume with the
  median volume. Lower values are better.

.. _iqms_dummy:

Number of *dummy* scans** (``dummy``)
  A number of volumes in the beginning of the
  fMRI timeseries identified as non-steady state.

.. topic:: References

  .. [Atkinson1997] Atkinson et al., *Automatic correction of motion artifacts
    in magnetic resonance images using an entropy
    focus criterion*, IEEE Trans Med Imag 16(6):903-910, 1997.
    doi:`10.1109/42.650886 <http://dx.doi.org/10.1109/42.650886>`_.

  .. [Friedman2008] Friedman, L et al., *Test--retest and between‐site reliability in a multicenter
    fMRI study*. Hum Brain Mapp, 29(8):958--972, 2008. doi:`10.1002/hbm.20440
    <http://dx.doi.org/10.1002/hbm.20440>`_.

  .. [Giannelli2010] Giannelli et al., *Characterization of Nyquist ghost in
    EPI-fMRI acquisition sequences implemented on two clinical 1.5 T MR scanner
    systems: effect of readout bandwidth and echo spacing*. J App Clin Med Phy,
    11(4). 2010.
    doi:`10.1120/jacmp.v11i4.3237 <http://dx.doi.org/10.1120/jacmp.v11i4.3237>`_.

  .. [Jenkinson2002] Jenkinson et al., *Improved Optimisation for the Robust and
    Accurate Linear Registration and Motion Correction of Brain Images*.
    NeuroImage, 17(2), 825-841, 2002.
    doi:`10.1006/nimg.2002.1132 <http://dx.doi.org/10.1006/nimg.2002.1132>`_.

  .. [Kruger2001] Krüger et al., *Physiological noise in oxygenation-sensitive
    magnetic resonance imaging*, Magn. Reson. Med. 46(4):631-637, 2001.
    doi:`10.1002/mrm.1240 <http://dx.doi.org/10.1002/mrm.1240>`_.

  .. [Nichols2013] Nichols, `Notes on Creating a Standardized Version of DVARS
    <http://www2.warwick.ac.uk/fac/sci/statistics/staff/academic-research/nichols/scripts/fsl/standardizeddvars.pdf>`_,
    2013.

  .. [Power2012] Power et al., *Spurious but systematic correlations in
    functional connectivity MRI networks arise from subject motion*,
    NeuroImage 59(3):2142-2154,
    2012, doi:`10.1016/j.neuroimage.2011.10.018
    <http://dx.doi.org/10.1016/j.neuroimage.2011.10.018>`_.

  .. [Saad2013] Saad et al. *Correcting Brain-Wide Correlation Differences
    in Resting-State FMRI*, Brain Conn 3(4):339-352,
    2013, doi:`10.1089/brain.2013.0156
    <http://dx.doi.org/10.1089/brain.2013.0156>`_.
"""
import os.path as op

import numpy as np

RAS_AXIS_ORDER = {"x": 0, "y": 1, "z": 2}


def gsr(epi_data, mask, direction="y", ref_file=None, out_file=None):
    """
    Compute the :abbr:`GSR (ghost to signal ratio)` [Giannelli2010]_.

    The procedure is as follows:

      #. Create a Nyquist ghost mask by circle-shifting the original mask by :math:`N/2`.

      #. Rotate by :math:`N/2`

      #. Remove the intersection with the original mask

      #. Generate a non-ghost background

      #. Calculate the :abbr:`GSR (ghost to signal ratio)`


    .. warning ::

      This should be used with EPI images for which the phase
      encoding direction is known.

    :param str epi_file: path to epi file
    :param str mask_file: path to brain mask
    :param str direction: the direction of phase encoding (x, y, all)
    :return: the computed gsr

    """
    direction = direction.lower()
    if direction[-1] not in ["x", "y", "all"]:
        raise Exception(
            "Unknown direction {}, should be one of x, -x, y, -y, all".format(direction)
        )

    if direction == "all":
        result = []
        for newdir in ["x", "y"]:
            ofile = None
            if out_file is not None:
                fname, ext = op.splitext(ofile)
                if ext == ".gz":
                    fname, ext2 = op.splitext(fname)
                    ext = ext2 + ext
                ofile = "{0}_{1}{2}".format(fname, newdir, ext)
            result += [gsr(epi_data, mask, newdir, ref_file=ref_file, out_file=ofile)]
        return result

    # Roll data of mask through the appropriate axis
    axis = RAS_AXIS_ORDER[direction]
    n2_mask = np.roll(mask, mask.shape[axis] // 2, axis=axis)

    # Step 3: remove from n2_mask pixels inside the brain
    n2_mask = n2_mask * (1 - mask)

    # Step 4: non-ghost background region is labeled as 2
    n2_mask = n2_mask + 2 * (1 - n2_mask - mask)

    # Step 5: signal is the entire foreground image
    ghost = np.mean(epi_data[n2_mask == 1]) - np.mean(epi_data[n2_mask == 2])
    signal = np.median(epi_data[n2_mask == 0])
    return float(ghost / signal)
