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
Measures based on noise measurements
====================================

.. _iqms_cjv:

- :py:func:`~mriqc.qc.anatomical.cjv` -- **coefficient of joint variation**
  (:abbr:`CJV (coefficient of joint variation)`):
  The ``cjv`` of GM and WM was proposed as objective function by [Ganzetti2016]_ for
  the optimization of :abbr:`INU (intensity non-uniformity)` correction algorithms.
  Higher values are related to the presence of heavy head motion and large
  :abbr:`INU (intensity non-uniformity)` artifacts. Lower values are better.

.. _iqms_cnr:

- :py:func:`~mriqc.qc.anatomical.cnr` -- **contrast-to-noise ratio**
  (:abbr:`CNR (contrast-to-noise ratio)`): The ``cnr`` [Magnota2006]_,
  is an extension of the :abbr:`SNR (signal-to-noise Ratio)` calculation
  to evaluate how separated the tissue distributions of GM and WM are.
  Higher values indicate better quality.

.. _iqms_snr:

- :py:func:`~mriqc.qc.anatomical.snr` -- **signal-to-noise ratio**
  (:abbr:`SNR (signal-to-noise ratio)`): calculated within the
  tissue mask.

.. _iqms_snrd:

- :py:func:`~mriqc.qc.anatomical.snr_dietrich`: **Dietrich's SNR**
  (:abbr:`SNRd (signal-to-noise ratio, Dietrich 2007)`) as proposed
  by [Dietrich2007]_, using the air background as reference.

.. _iqms_qi2:

- :py:func:`~mriqc.qc.anatomical.art_qi2`: **Mortamet's quality index 2**
  (:abbr:`QI2 (quality index 2)`) is a calculation of the goodness-of-fit
  of a :math:`\chi^2` distribution on the air mask,
  once the artifactual intensities detected for computing
  the :abbr:`QI1 (quality index 1)` index have been removed [Mortamet2009]_.
  Lower values are better.

Measures based on information theory
------------------------------------

.. _iqms_efc:

- :py:func:`~mriqc.qc.anatomical.efc`:
  The :abbr:`EFC (Entropy Focus Criterion)`
  [Atkinson1997]_ uses the Shannon entropy of voxel intensities as
  an indication of ghosting and blurring induced by head motion.
  Lower values are better.

  The original equation is normalized by the maximum entropy, so that the
  :abbr:`EFC (Entropy Focus Criterion)` can be compared across images with
  different dimensions.

.. _iqms_fber:

- :py:func:`~mriqc.qc.anatomical.fber`:
  The :abbr:`FBER (Foreground-Background Energy Ratio)` [Shehzad2015]_,
  defined as the mean energy of image values within the head relative
  to outside the head [QAP-measures]_.
  Higher values are better.

Measures targeting specific artifacts
-------------------------------------

.. _iqms_inu:

- **inu_\*** (*nipype interface to N4ITK*): summary statistics (max, min and median)
  of the :abbr:`INU (intensity non-uniformity)` field (bias field) as extracted
  by the N4ITK algorithm [Tustison2010]_. Values closer to 1.0 are better, values
  further from zero indicate greater RF field inhomogeneity.

.. _iqms_qi:

- :py:func:`~mriqc.qc.anatomical.art_qi1`:
  Detect artifacts in the image using the method described in [Mortamet2009]_.
  The :abbr:`QI1 (quality index 1)` is the proportion of voxels with intensity
  corrupted by artifacts normalized by the number of voxels in the background.
  Lower values are better.

  .. figure:: ../resources/mortamet-mrm2009.png

    The workflow to compute the artifact detection from [Mortamet2009]_.

.. _iqms_wm2max:

- :py:func:`~mriqc.qc.anatomical.wm2max`:
  The white-matter to maximum intensity ratio is the median intensity
  within the WM mask over the 95% percentile of the full intensity
  distribution, that captures the existence of long tails due to
  hyper-intensity of the carotid vessels and fat. Values
  should be around the interval [0.6, 0.8].


Other measures
^^^^^^^^^^^^^^

.. _iqms_fwhm:

- **fwhm** (*nipype interface to AFNI*): The :abbr:`FWHM (full-width half maximum)` of
  the spatial distribution of the image intensity values in units of voxels [Forman1995]_.
  Lower values are better, higher values indicate a blurrier image. Uses the gaussian
  width estimator filter implemented in AFNI's ``3dFWHMx``:

  .. math ::

      \text{FWHM} = \sqrt{-{\left[4 \ln{(1-\frac{\sigma^2_{X^m_{i+1,j}-X^m_{i,j}}}
      {2\sigma^2_{X^m_{i,j}}}})\right]}^{-1}}


.. _iqms_icvs:

- :py:func:`~mriqc.qc.anatomical.volume_fraction` (**icvs_\***):
  the
  :abbr:`ICV (intracranial volume)` fractions of :abbr:`CSF (cerebrospinal fluid)`,
  :abbr:`GM (gray-matter)` and :abbr:`WM (white-matter)`. They should move within
  a normative range.

.. _iqms_rpve:

- :py:func:`~mriqc.qc.anatomical.rpve` (**rpve_\***): the
  :abbr:`rPVe (residual partial voluming error)` of :abbr:`CSF (cerebrospinal fluid)`,
  :abbr:`GM (gray-matter)` and :abbr:`WM (white-matter)`. Lower values are better.

.. _iqms_summary:

- :py:func:`~mriqc.qc.anatomical.summary_stats` (**summary_\*_\***):
  Mean, median, median absolute deviation (mad), standard deviation, kurtosis,
  5% percentile, 95% percentile and number of voxels of the distribution of background,
  :abbr:`CSF (cerebrospinal fluid)`, :abbr:`GM (gray-matter)` and :abbr:`WM (white-matter)`.

.. _iqms_tpm:

- **overlap_\*_\***:
  The overlap of the :abbr:`TPMs (tissue probability maps)` estimated from the image and
  the corresponding maps from the ICBM nonlinear-asymmetric 2009c template. Higher
  values are better.

  .. math ::

      \text{JI}^k = \frac{\sum_i \min{(\text{TPM}^k_i, \text{MNI}^k_i)}}
      {\sum_i \max{(\text{TPM}^k_i, \text{MNI}^k_i)}}


.. topic:: References

  .. [Dietrich2007] Dietrich et al., *Measurement of SNRs in MR images: influence
    of multichannel coils, parallel imaging and reconstruction filters*, JMRI 26(2):375--385.
    2007. doi:`10.1002/jmri.20969 <http://dx.doi.org/10.1002/jmri.20969>`_.

  .. [Ganzetti2016] Ganzetti et al., *Intensity inhomogeneity correction of structural MR images:
    a data-driven approach to define input algorithm parameters*. Front Neuroinform 10:10. 2016.
    doi:`10.3389/fninf.2016.00010 <https://doi.org/10.3389/fninf.2016.00010>`_.

  .. [Magnota2006] Magnotta, VA., & Friedman, L., *Measurement of signal-to-noise
    and contrast-to-noise in the fBIRN multicenter imaging study*.
    J Dig Imag 19(2):140-147, 2006. doi:`10.1007/s10278-006-0264-x
    <http://dx.doi.org/10.1007/s10278-006-0264-x>`_.

  .. [Mortamet2009] Mortamet B et al., *Automatic quality assessment in
    structural brain magnetic resonance imaging*, Mag Res Med 62(2):365-372,
    2009. doi:`10.1002/mrm.21992 <http://dx.doi.org/10.1002/mrm.21992>`_.

  .. [Tustison2010] Tustison NJ et al., *N4ITK: improved N3 bias correction*,
    IEEE Trans Med Imag, 29(6):1310-20,
    2010. doi:`10.1109/TMI.2010.2046908 <http://dx.doi.org/10.1109/TMI.2010.2046908>`_.

  .. [Shehzad2015] Shehzad Z et al., *The Preprocessed Connectomes Project
     Quality Assessment Protocol - a resource for measuring the quality of MRI data*,
     Front. Neurosci. Conference Abstract: Neuroinformatics 2015.
     doi:`10.3389/conf.fnins.2015.91.00047 <https://doi.org/10.3389/conf.fnins.2015.91.00047>`_.

  .. [Forman1995] Forman SD et al., *Improved assessment of significant activation in functional
     magnetic resonance imaging (fMRI): use of a cluster-size threshold*,
     Magn. Reson. Med. 33 (5), 636–647, 1995.
     doi:`10.1002/mrm.1910330508 <https://doi.org/10.1002/mrm.1910330508>`_.
"""
import os.path as op
from math import sqrt

import numpy as np
import scipy.ndimage as nd
from scipy.stats import kurtosis  # pylint: disable=E0611
from mriqc import config


DIETRICH_FACTOR = 0.6551364  # 1.0 / sqrt(2 / (4 - pi))
FSL_FAST_LABELS = {"csf": 1, "gm": 2, "wm": 3, "bg": 0}


def snr(mu_fg, sigma_fg, n):
    r"""
    Calculate the :abbr:`SNR (Signal-to-Noise Ratio)`.
    The estimation may be provided with only one foreground region in
    which the noise is computed as follows:

    .. math::

        \text{SNR} = \frac{\mu_F}{\sigma_F\sqrt{n/(n-1)}},

    where :math:`\mu_F` is the mean intensity of the foreground and
    :math:`\sigma_F` is the standard deviation of the same region.

    :param float mu_fg: mean of foreground.
    :param float sigma_fg: standard deviation of foreground.
    :param int n: number of voxels in foreground mask.

    :return: the computed SNR

    """
    return float(mu_fg / (sigma_fg * sqrt(n / (n - 1))))


def snr_dietrich(mu_fg, mad_air=0.0, sigma_air=1.0):
    r"""
    Calculate the :abbr:`SNR (Signal-to-Noise Ratio)`.

    This must be an air mask around the head, and it should not contain artifacts.
    The computation is done following the eq. A.12 of [Dietrich2007]_, which
    includes a correction factor in the estimation of the standard deviation of
    air and its Rayleigh distribution:

    .. math::

        \text{SNR} = \frac{\mu_F}{\sqrt{\frac{2}{4-\pi}}\,\sigma_\text{air}}.


    :param float mu_fg: mean of foreground.
    :param float sigma_air: standard deviation of the air surrounding the head ("hat" mask).

    :return: the computed SNR for the foreground segmentation

    """
    if mad_air > 1.0:
        return float(DIETRICH_FACTOR * mu_fg / mad_air)

    config.loggers.interface.warning(
        "Estimated signal variation in the background was too small "
        f"(MAD={mad_air}, sigma={sigma_air})",
    )
    return float(DIETRICH_FACTOR * mu_fg / sigma_air) if sigma_air > 1e-3 else -1.0


def cnr(mu_wm, mu_gm, sigma_air, sigma_wm, sigma_gm):
    r"""
    Calculate the :abbr:`CNR (Contrast-to-Noise Ratio)` [Magnota2006]_.
    Higher values are better.

    .. math::

        \text{CNR} = \frac{|\mu_\text{GM} - \mu_\text{WM} |}{\sqrt{\sigma_B^2 +
        \sigma_\text{WM}^2 + \sigma_\text{GM}^2}},

    where :math:`\sigma_B` is the standard deviation of the noise distribution within
    the air (background) mask.


    :param float mu_wm: mean of signal within white-matter mask.
    :param float mu_gm: mean of signal within gray-matter mask.
    :param float sigma_air: standard deviation of the air surrounding the head ("hat" mask).
    :param float sigma_wm: standard deviation within white-matter mask.
    :param float sigma_gm: standard within gray-matter mask.

    :return: the computed CNR

    """
    return float(
        abs(mu_wm - mu_gm) / sqrt(sigma_air**2 + sigma_gm**2 + sigma_wm**2)
    )


def cjv(mu_wm, mu_gm, sigma_wm, sigma_gm):
    r"""
    Calculate the :abbr:`CJV (coefficient of joint variation)`, a measure
    related to :abbr:`SNR (Signal-to-Noise Ratio)` and
    :abbr:`CNR (Contrast-to-Noise Ratio)` that is presented as a proxy for
    the :abbr:`INU (intensity non-uniformity)` artifact [Ganzetti2016]_.
    Lower is better.

    .. math::

        \text{CJV} = \frac{\sigma_\text{WM} + \sigma_\text{GM}}{|\mu_\text{WM} - \mu_\text{GM}|}.

    :param float mu_wm: mean of signal within white-matter mask.
    :param float mu_gm: mean of signal within gray-matter mask.
    :param float sigma_wm: standard deviation of signal within white-matter mask.
    :param float sigma_gm: standard deviation of signal within gray-matter mask.

    :return: the computed CJV


    """
    return float((sigma_wm + sigma_gm) / abs(mu_wm - mu_gm))


def fber(img, headmask, rotmask=None):
    r"""
    Calculate the :abbr:`FBER (Foreground-Background Energy Ratio)` [Shehzad2015]_,
    defined as the mean energy of image values within the head relative
    to outside the head.
    Higher values are better, and an FBER=-1.0 indicates that there is no signal
    outside the head mask (e.g., a skull-stripped dataset).

    .. math::

        \text{FBER} = \frac{E[|F|^2]}{E[|B|^2]}


    :param numpy.ndarray img: input data
    :param numpy.ndarray headmask: a mask of the head (including skull, skin, etc.)
    :param numpy.ndarray rotmask: a mask of empty voxels inserted after a rotation of
      data

    """

    fg_mu = np.median(np.abs(img[headmask > 0]) ** 2)

    airmask = np.ones_like(headmask, dtype=np.uint8)
    airmask[headmask > 0] = 0
    if rotmask is not None:
        airmask[rotmask > 0] = 0
    bg_mu = np.median(np.abs(img[airmask == 1]) ** 2)
    if bg_mu < 1.0e-3:
        return -1.0
    return float(fg_mu / bg_mu)


def efc(img, framemask=None):
    r"""
    Calculate the :abbr:`EFC (Entropy Focus Criterion)` [Atkinson1997]_.
    Uses the Shannon entropy of voxel intensities as an indication of ghosting
    and blurring induced by head motion. A range of low values is better,
    with EFC = 0 for all the energy concentrated in one pixel.

    .. math::

        \text{E} = - \sum_{j=1}^N \frac{x_j}{x_\text{max}}
        \ln \left[\frac{x_j}{x_\text{max}}\right]

    with :math:`x_\text{max} = \sqrt{\sum_{j=1}^N x^2_j}`.

    The original equation is normalized by the maximum entropy, so that the
    :abbr:`EFC (Entropy Focus Criterion)` can be compared across images with
    different dimensions:

    .. math::

        \text{EFC} = \left( \frac{N}{\sqrt{N}} \, \log{\sqrt{N}^{-1}} \right) \text{E}

    :param numpy.ndarray img: input data
    :param numpy.ndarray framemask: a mask of empty voxels inserted after a rotation of
      data

    """

    if framemask is None:
        framemask = np.zeros_like(img, dtype=np.uint8)

    n_vox = np.sum(1 - framemask)
    # Calculate the maximum value of the EFC (which occurs any time all
    # voxels have the same value)
    efc_max = 1.0 * n_vox * (1.0 / np.sqrt(n_vox)) * np.log(1.0 / np.sqrt(n_vox))

    # Calculate the total image energy
    b_max = np.sqrt((img[framemask == 0] ** 2).sum())

    # Calculate EFC (add 1e-16 to the image data to keep log happy)
    return float(
        (1.0 / efc_max)
        * np.sum(
            (img[framemask == 0] / b_max)
            * np.log((img[framemask == 0] + 1e-16) / b_max)
        )
    )


def wm2max(img, mu_wm):
    r"""
    Calculate the :abbr:`WM2MAX (white-matter-to-max ratio)`,
    defined as the maximum intensity found in the volume w.r.t. the
    mean value of the white matter tissue. Values close to 1.0 are
    better:

    .. math ::

        \text{WM2MAX} = \frac{\mu_\text{WM}}{P_{99.95}(X)}

    """
    return float(mu_wm / np.percentile(img.reshape(-1), 99.95))


def art_qi1(airmask, artmask):
    r"""
    Detect artifacts in the image using the method described in [Mortamet2009]_.
    Calculates :math:`\text{QI}_1`, as the proportion of voxels with intensity
    corrupted by artifacts normalized by the number of voxels in the "*hat*"
    mask (i.e., the background region above the nasio-occipital plane):

    .. math ::

        \text{QI}_1 = \frac{1}{N} \sum\limits_{x\in X_\text{art}} 1

    Near-zero values are better.
    If :math:`\text{QI}_1 = -1`, then the "*hat*" mask (background) was empty
    and the dataset is likely a skull-stripped image or has been heavily
    post-processed.

    :param numpy.ndarray airmask: input air mask, without artifacts
    :param numpy.ndarray artmask: input artifacts mask

    """
    if airmask.sum() < 1:
        return -1.0

    # Count the ratio between artifacts and the total voxels in "hat" mask
    return float(artmask.sum() / (airmask.sum() + artmask.sum()))


def art_qi2(img, airmask, min_voxels=int(1e3), max_voxels=int(3e5), save_plot=True):
    r"""
    Calculates :math:`\text{QI}_2`, based on the goodness-of-fit of a centered
    :math:`\chi^2` distribution onto the intensity distribution of
    non-artifactual background (within the "hat" mask):


    .. math ::

        \chi^2_n = \frac{2}{(\sigma \sqrt{2})^{2n} \, (n - 1)!}x^{2n - 1}\, e^{-\frac{x}{2}}

    where :math:`n` is the number of coil elements.

    :param numpy.ndarray img: input data
    :param numpy.ndarray airmask: input air mask without artifacts

    """

    from nireports.reportlets.nuisance import plot_qi2
    from scipy.stats import chi2
    from sklearn.neighbors import KernelDensity

    # S. Ogawa was born
    np.random.seed(1191935)

    data = img[airmask > 0]
    data = data[data > 0]

    # Write out figure of the fitting
    out_file = op.abspath("error.svg")
    with open(out_file, "w") as ofh:
        ofh.write("<p>Background noise fitting could not be plotted.</p>")

    if len(data) < min_voxels:
        return 0.0, out_file

    modelx = data if len(data) < max_voxels else np.random.choice(data, size=max_voxels)

    x_grid = np.linspace(0.0, np.percentile(data, 99), 1000)

    # Estimate data pdf with KDE on a random subsample
    kde_skl = KernelDensity(
        bandwidth=0.05 * np.percentile(data, 98), kernel="gaussian"
    ).fit(modelx[:, np.newaxis])
    kde = np.exp(kde_skl.score_samples(x_grid[:, np.newaxis]))

    # Find cutoff
    kdethi = np.argmax(kde[::-1] > kde.max() * 0.5)

    # Fit X^2
    param = chi2.fit(modelx[modelx < np.percentile(data, 95)], 32)
    chi_pdf = chi2.pdf(x_grid, *param[:-2], loc=param[-2], scale=param[-1])

    # Compute goodness-of-fit (gof)
    gof = float(np.abs(kde[-kdethi:] - chi_pdf[-kdethi:]).mean())
    if save_plot:
        out_file = plot_qi2(x_grid, kde, chi_pdf, modelx, kdethi)

    return gof, out_file


def volume_fraction(pvms):
    r"""
    Computes the :abbr:`ICV (intracranial volume)` fractions
    corresponding to the (partial volume maps).

    .. math ::

        \text{ICV}^k = \frac{\sum_i p^k_i}{\sum\limits_{x \in X_\text{brain}} 1}

    :param list pvms: list of :code:`numpy.ndarray` of partial volume maps.

    """
    tissue_vfs = {}
    total = 0
    for k, lid in list(FSL_FAST_LABELS.items()):
        if lid == 0:
            continue
        tissue_vfs[k] = pvms[lid - 1].sum()
        total += tissue_vfs[k]

    for k in list(tissue_vfs.keys()):
        tissue_vfs[k] /= total
    return {k: float(v) for k, v in list(tissue_vfs.items())}


def rpve(pvms, seg):
    """
    Computes the :abbr:`rPVe (residual partial voluming error)`
    of each tissue class.

    .. math ::

        \\text{rPVE}^k = \\frac{1}{N} \\left[ \\sum\\limits_{p^k_i \
\\in [0.5, P_{98}]} p^k_i + \\sum\\limits_{p^k_i \\in [P_{2}, 0.5)} 1 - p^k_i \\right]

    """

    pvfs = {}
    for k, lid in list(FSL_FAST_LABELS.items()):
        if lid == 0:
            continue
        pvmap = pvms[lid - 1]
        pvmap[pvmap < 0.0] = 0.0
        pvmap[pvmap >= 1.0] = 1.0
        totalvol = np.sum(pvmap > 0.0)
        upth = np.percentile(pvmap[pvmap > 0], 98)
        loth = np.percentile(pvmap[pvmap > 0], 2)
        pvmap[pvmap < loth] = 0
        pvmap[pvmap > upth] = 0
        pvfs[k] = (
            pvmap[pvmap > 0.5].sum() + (1.0 - pvmap[pvmap <= 0.5]).sum()
        ) / totalvol
    return {k: float(v) for k, v in list(pvfs.items())}


def summary_stats(img, pvms, airmask=None, erode=True):
    r"""
    Estimates the mean, the median, the standard deviation,
    the kurtosis,the median absolute deviation (mad), the 95\%
    and the 5\% percentiles and the number of voxels (summary\_\*\_n)
    of each tissue distribution.

    .. warning ::

        Sometimes (with datasets that have been partially processed), the air
        mask will be empty. In those cases, the background stats will be zero
        for the mean, median, percentiles and kurtosis, the sum of voxels in
        the other remaining labels for ``n``, and finally the MAD and the
        :math:`\sigma` will be calculated as:

        .. math ::

            \sigma_\text{BG} = \sqrt{\sum \sigma_\text{i}^2}


    """
    from statsmodels.robust.scale import mad

    from .. import config

    # Check type of input masks
    dims = np.squeeze(np.array(pvms)).ndim
    if dims == 4:
        # If pvms is from FSL FAST, create the bg mask
        stats_pvms = [np.zeros_like(img)] + pvms
    elif dims == 3:
        stats_pvms = [np.ones_like(pvms) - pvms, pvms]
    else:
        raise RuntimeError(
            "Incorrect image dimensions ({0:d})".format(np.array(pvms).ndim)
        )

    if airmask is not None:
        stats_pvms[0] = airmask

    labels = list(FSL_FAST_LABELS.items())
    if len(stats_pvms) == 2:
        labels = list(zip(["bg", "fg"], list(range(2))))

    output = {}
    for k, lid in labels:
        mask = np.zeros_like(img, dtype=np.uint8)
        mask[stats_pvms[lid] > 0.85] = 1

        if erode:
            struct = nd.generate_binary_structure(3, 2)
            mask = nd.binary_erosion(mask, structure=struct).astype(np.uint8)

        nvox = float(mask.sum())
        if nvox < 1e3:
            config.loggers.interface.warning(
                'calculating summary stats of label "%s" in a very small '
                "mask (%d voxels)",
                k,
                int(nvox),
            )
            if k == "bg":
                output["bg"] = {
                    "mean": 0.0,
                    "median": 0.0,
                    "p95": 0.0,
                    "p05": 0.0,
                    "k": 0.0,
                    "stdv": 0.0,
                    "mad": 0.0,
                    "n": nvox,
                }
                continue

        output[k] = {
            "mean": float(img[mask == 1].mean()),
            "stdv": float(img[mask == 1].std()),
            "median": float(np.median(img[mask == 1])),
            "mad": float(mad(img[mask == 1])),
            "p95": float(np.percentile(img[mask == 1], 95)),
            "p05": float(np.percentile(img[mask == 1], 5)),
            "k": float(kurtosis(img[mask == 1])),
            "n": nvox,
        }

    return output


def _prepare_mask(mask, label, erode=True):
    fgmask = mask.copy()

    if np.issubdtype(fgmask.dtype, np.integer):
        if isinstance(label, (str, bytes)):
            label = FSL_FAST_LABELS[label]

        fgmask[fgmask != label] = 0
        fgmask[fgmask == label] = 1
    else:
        fgmask[fgmask > 0.95] = 1.0
        fgmask[fgmask < 1.0] = 0

    if erode:
        # Create a structural element to be used in an opening operation.
        struct = nd.generate_binary_structure(3, 2)
        # Perform an opening operation on the background data.
        fgmask = nd.binary_opening(fgmask, structure=struct).astype(np.uint8)

    return fgmask
