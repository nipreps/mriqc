#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
# pylint: disable=no-member
#
# @Author: oesteban
# @Date:   2016-02-23 19:25:39
# @Email:  code@oscaresteban.es
# @Last Modified by:   oesteban
# @Last Modified time: 2017-03-09 17:36:58
"""

Measures for the structural information
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Definitions are given in the
:ref:`summary of structural IQMs <iqms_t1w>`.

- **Entropy-focus criterion** (:py:func:`~mriqc.qc.anatomical.efc`).
- **Foreground-Background energy ratio** (:py:func:`~mriqc.qc.anatomical.fber`,  [Shehzad2015]_).
- **Full-width half maximum smoothness** (``fwhm_*``).
- **Signal-to-noise ratio** (:py:func:`~mriqc.qc.anatomical.snr`).
- **Summary statistics** (``summary_*_*``).


Measures for the temporal information
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- **DVARS** - D referring to temporal derivative of timecourses, VARS referring to
  RMS variance over voxels (``dvars``), calculated
  `with nipype <http://nipype.readthedocs.io/en/latest/interfaces/generated/\
nipype.algorithms.confounds.html#computedvars>`_ before motion correction.
- **Ghost to Signal Ratio** (:py:func:`~mriqc.qc.functional.gsr`, ``ghost_*``:
  along the two possible phase-encoding axes **x**, **y**.
- **Global Correlation** (:py:func:`~mriqc.qc.functional.gcor`, ``gcor``).
- **Temporal SNR** (:abbr:`tSNR (temporal SNR)`, ``tsnr``) is the median value
  of the `tSNR map <http://nipype.readthedocs.io/en/latest/interfaces/generated/\
nipype.algorithms.confounds.html#tsnr>`_.

Measures for artifacts and other
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- **Framewise Displacement** (``mean_fd``, [Power2012]_).
- **Number of timepoints above FD theshold** (``num_fd``): the threshold is defined
  at 0.20mm, so :abbr:`FD (frame displacement)` :math:`> 0.20mm`
- **Percent of ``num_fd`` w.r.t. the timeseries**.
- **Outlier fraction** (``outlier``) - Mean fraction of outliers per fMRI volume
  as given by AFNI.
- **Quality index** (``quality``) - Mean quality index as computed by AFNI.

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

  .. [Nichols2013] Nichols, `Notes on Creating a Standardized Version of DVARS
      <http://www2.warwick.ac.uk/fac/sci/statistics/staff/academic-research/nichols/scripts/fsl/standardizeddvars.pdf>`_, 2013.

  .. [Power2012] Power et al., *Spurious but systematic correlations in
    functional connectivity MRI networks arise from subject motion*,
    NeuroImage 59(3):2142-2154,
    2012, doi:`10.1016/j.neuroimage.2011.10.018
    <http://dx.doi.org/10.1016/j.neuroimage.2011.10.018>`_.

  .. [Saad2013] Saad et al. *Correcting Brain-Wide Correlation Differences
     in Resting-State FMRI*, Brain Conn 3(4):339-352,
     2013, doi:`10.1089/brain.2013.0156
     <http://dx.doi.org/10.1089/brain.2013.0156>`_.


mriqc.qc.functional module
^^^^^^^^^^^^^^^^^^^^^^^^^^

"""
from __future__ import print_function, division, absolute_import, unicode_literals
import os.path as op
import numpy as np
import nibabel as nb

RAS_AXIS_ORDER = {'x': 0, 'y': 1, 'z': 2}

def gsr(epi_data, mask, direction="y", ref_file=None, out_file=None):
    """
    Computes the :abbr:`GSR (ghost to signal ratio)` [Giannelli2010]_. The
    procedure is as follows:

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
    if direction[-1] not in ['x', 'y', 'all']:
        raise Exception("Unknown direction {}, should be one of x, -x, y, -y, all".format(
            direction))

    if direction == 'all':
        result = []
        for newdir in ['x', 'y']:
            ofile = None
            if out_file is not None:
                fname, ext = op.splitext(ofile)
                if ext == '.gz':
                    fname, ext2 = op.splitext(fname)
                    ext = ext2 + ext
                ofile = '{0}_{1}{2}'.format(fname, newdir, ext)
            result += [gsr(epi_data, mask, newdir,
                           ref_file=ref_file, out_file=ofile)]
        return result

    # Roll data of mask through the appropriate axis
    axis = RAS_AXIS_ORDER[direction]
    n2_mask = np.roll(mask, mask.shape[axis]//2, axis=axis)

    # Step 3: remove from n2_mask pixels inside the brain
    n2_mask = n2_mask * (1-mask)

    # Step 4: non-ghost background region is labeled as 2
    n2_mask = n2_mask + 2 * (1 - n2_mask - mask)

    # Step 5: signal is the entire foreground image
    ghost = np.mean(epi_data[n2_mask == 1]) - np.mean(epi_data[n2_mask == 2])
    signal = np.median(epi_data[n2_mask == 0])
    return float(ghost/signal)


def gcor(func, mask=None):
    """
    Compute the :abbr:`GCOR (global correlation)` [Saad2013]_.

    :param numpy.ndarray func: input fMRI dataset, after motion correction
    :param numpy.ndarray mask: 3D brain mask
    :return: the computed GCOR value

    """
    import numpy as np
    from statsmodels.robust.scale import mad

    # Reshape to N voxels x T timepoints
    func_v = func.reshape(-1, func.shape[-1])

    if mask is not None:
        func_v = np.squeeze(func_v.take(np.where(mask.reshape(-1) > 0), axis=0))

    func_sigma = mad(func_v, axis=1)
    mask = np.zeros_like(func_sigma)
    mask[func_sigma > 1.e-5] = 1

    # Remove zero-variance voxels across time axis
    func_v = np.squeeze(func_v.take(np.where(mask > 0), axis=0))
    func_sigma = func_sigma[mask > 0]
    func_mean = np.median(func_v, axis=1)

    zscored = func_v - func_mean[..., np.newaxis]
    zscored /= func_sigma[..., np.newaxis]

    # avg_ts is an N timepoints x 1 vector
    avg_ts = zscored.mean(axis=0)
    return float(avg_ts.T.dot(avg_ts) / len(avg_ts))
