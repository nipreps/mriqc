
.. _t1w:


T1-weighed images
-----------------

Measures based on noise measurements
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- :py:func:`~mriqc.qc.anatomical.cjv`: 
  The :abbr:`CJV (coefficient of joint variation)` of GM and WM was proposed as objective function by [Ganzetti2016]_ for the optimization of :abbr:`INU (intensity non-uniformity)` correction algorithms. Higher values are related to the presence of heavy head motion and large :abbr:`INU (intensity non-uniformity)` artifacts. Lower values are better.

- :py:func:`~mriqc.qc.anatomical.cnr`:
  The :abbr:`CNR (contrast-to-noise Ratio)` [Magnota2006]_, is an extension of the :abbr:`SNR (signal-to-noise Ratio)` calculation to evaluate how separated the tissue distributions of GM and WM are. Higher values indicate better quality.

- :py:func:`~mriqc.qc.anatomical.snr_dietrich`: 
  MRIQC includes the :abbr:`SNR (signal-to-noise Ratio)` as proposed
  by [Dietrich2007]_, using the air background as reference.
  Additionally, for images that have undergone some noise reduction
  processing, or the more complex noise realizations of current
  parallel acquisitions, a simplified calculation using the within
  tissue variance is also provided

- :py:func:`~mriqc.qc.anatomical.art_qi2`: The 
  :abbr:`QI2 (quality index 2)` of [Mortamet2009]_ is a calculation
  of the goodness-of-fit of a :math:`\chi^2` distribution on the 
  air mask, once the artifactual intensities detected for computing
  the :abbr:`QI1 (quality index 1)` index have been removed.

Measures based on information theory
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- :py:func:`~mriqc.qc.anatomical.efc`:
  The :abbr:`EFC (Entropy Focus Criterion)`
  [Atkinson1997]_ uses the Shannon entropy of voxel intensities as 
  an indication of ghosting and blurring induced by head motion.
  Lower values are better.

  The original equation is normalized by the maximum entropy, so that the
  :abbr:`EFC (Entropy Focus Criterion)` can be compared across images with
  different dimensions.

- :py:func:`~mriqc.qc.anatomical.fber`:
  The :abbr:`FBER (Foreground-Background Energy Ratio)`,
  defined as the mean energy of image values within the head relative to outside the head [QAP-measures]_.
  Higher values are better.

Measures targeting specific artifacts
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- **inu_\*** (*nipype interface to N4ITK*): summary statistics (max, min and median)
  of the :abbr:`INU (intensity non-uniformity)` field as extracted by the N4ITK algorithm
  [Tustison2010]_. Values closer to 1.0 are better.

- :py:func:`~mriqc.qc.anatomical.art_qi1`:
  Detect artifacts in the image using the method described in [Mortamet2009]_.
  The :abbr:`QI1 (quality index 1)` is the proportion of voxels with intensity corrupted by artifacts
  normalized by the number of voxels in the background. Lower values are better.

  Optionally, it also calculates **qi2**, the distance between the distribution
  of noise voxel (non-artifact background voxels) intensities, and a
  Rician distribution.

  .. figure:: ../resources/mortamet-mrm2009.png

    The workflow to compute the artifact detection from [Mortamet2009]_.

- :py:func:`~mriqc.qc.anatomical.wm2max`:
  The white-matter to maximum intensity ratio is the median intensity
  within the WM mask over the 95% percentile of the full intensity
  distribution, that captures the existence of long tails due to
  hyper-intensity of the carotid vessels and fat. Values
  should be around the interval [0.6, 0.8]


Other measures
^^^^^^^^^^^^^^

- **fwhm** (*nipype interface to AFNI*): The :abbr:`FWHM (full-width half maximum)` of 
  the spatial distribution of the image intensity values in units of voxels [Friedman2008]_.
  Lower values are better

- :py:func:`~mriqc.qc.anatomical.volume_fractions` (**icvs_\***):
  the
  :abbr:`ICV (intracranial volume)` fractions of :abbr:`CSF (cerebrospinal fluid)`,
  :abbr:`GM (gray-matter)` and :abbr:`WM (white-matter)`. They should move within
  a normative range.

- :py:func:`~mriqc.qc.anatomical.rpve` (**rpve_\***): the
  :abbr:`rPVe (residual partial voluming error)` of :abbr:`CSF (cerebrospinal fluid)`,
  :abbr:`GM (gray-matter)` and :abbr:`WM (white-matter)`. Lower values are better.

- :py:func:`~mriqc.qc.anatomical.summary_stats` (**summary_\*_\***):
  Mean, standard deviation, 5% percentile and 95% percentile of the distribution
  of background, :abbr:`CSF (cerebrospinal fluid)`, :abbr:`GM (gray-matter)` and
  :abbr:`WM (white-matter)`.

- **overlap_\*_\***: 
  The overlap of the :abbr:`TPMs (tissue probability maps)` estimated from the image and the corresponding maps from the ICBM nonlinear-asymmetric 2009c template.


.. topic:: References

  .. [Dietrich2007] Dietrich et al., *Measurement of SNRs in MR images: influence
    of multichannel coils, parallel imaging and reconstruction filters*, JMRI 26(2):375--385.
    2007. doi:`10.1002/jmri.20969 <http://dx.doi.org/10.1002/jmri.20969>`_.

  .. [Ganzetti2016] Ganzetti et al., *Intensity inhomogeneity correction of structural MR images:
    a data-driven approach to define input algorithm parameters*. Front Neuroinform 10:10. 2016.
    doi:`10.3389/finf.201600010 <http://dx.doi.org/10.3389/finf.201600010>`_.

  .. [Magnota2006] Magnotta, VA., & Friedman, L., *Measurement of signal-to-noise
    and contrast-to-noise in the fBIRN multicenter imaging study*. 
    J Dig Imag 19(2):140-147, 2006. doi:`10.1007/s10278-006-0264-x
    <http://dx.doi.org/10.1007/s10278-006-0264-x>`_.

  .. [Mortamet2009] Mortamet B et al., *Automatic quality assessment in
    structural brain magnetic resonance imaging*, Mag Res Med 62(2):365-372,
    2009. doi:`10.1002/mrm.21992 <http://dx.doi.org/10.1002/mrm.21992>`_.

  .. [Tustison2010] Tustison NJ et al., *N4ITK: improved N3 bias correction*, IEEE Trans Med Imag, 29(6):1310-20,
    2010. doi:`10.1109/TMI.2010.2046908 <http://dx.doi.org/10.1109/TMI.2010.2046908>`_