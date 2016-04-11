MRIQC measures
==============

Structural images (:abbr:`sMRI (structural MRI)`)
-------------------------------------------------

The :abbr:`IQMs (image quality metrics)` computed are as follows:

- **fwhm** (*nipype interface to AFNI*): The :abbr:`FWHM (full-width half maximum)` of 
  the spatial distribution of the image intensity values in units of voxels [Friedman2008]_.
  Lower values are better

- **snr** (:py:func:`~mriqc.qc.anatomical.snr`): :abbr:`SNR (Signal-to-Noise Ratio)` as the mean
  of image values within each tissue type divided by the standard deviation of the image values 
  within air (i.e., outside the head) [Magnota2006]_. Higher values are better:

    .. math::

        \text{SNR} = \frac{\mu_F}{\sigma_\text{air}}


- **cnr** (:py:func:`~mriqc.qc.anatomical.cnr`): :abbr:`CNR (Contrast-to-Noise Ratio)` 
  [Magnota2006]_, high values are better:

    .. math::

        \text{CNR} = \frac{|\mu_\text{GM} - \mu_\text{WM} |}{\sigma_\text{air}},

    where :math:`\sigma_\text{air}` is the standard deviation of the noise distribution within
    the air (background) mask.

- **fber** (:py:func:`~mriqc.qc.anatomical.fber`): :abbr:`FBER (Foreground-Background Energy Ratio)`,
  defined as the mean energy of image values within the head relative to outside the head [QAP-measures]_.
  Higher values are better.

    .. math::

        \text{FBER} = \frac{E[|S|^2]}{E[|N_\text{air}|^2]}.

- **efc** (:py:func:`~mriqc.qc.anatomical.efc`): the :abbr:`EFC (Entropy Focus Criterion)`
  [Atkinson1997]_ uses the Shannon entropy of voxel intensities as an indication of ghosting
  and blurring induced by head motion. Lower values are better.

  The original equation is normalized by the maximum entropy, so that the
  :abbr:`EFC (Entropy Focus Criterion)` can be compared across images with
  different dimensions.


- **qi1** and **qi2** (:py:func:`~mriqc.qc.anatomical.artifact`):
  Detect artifacts in the image using the method described in [Mortamet2009]_.
  The **q1** is the proportion of voxels with intensity corrupted by artifacts
  normalized by the number of voxels in the background. Lower values are better.

  Optionally, it also calculates **qi2**, the distance between the distribution
  of noise voxel (non-artifact background voxels) intensities, and a
  Rician distribution.

  .. figure:: resources/mortamet-mrm2009.png

    The workflow to compute the artifact detection from [Mortamet2009]_.


- **icvs_\*** (:py:func:`~mriqc.qc.anatomical.volume_fractions`): the
  :abbr:`ICV (intracranial volume)` fractions of :abbr:`CSF (cerebrospinal fluid)`,
  :abbr:`GM (gray-matter)` and :abbr:`WM (white-matter)`. They should move within
  a normative range.

- **rpve_\*** (:py:func:`~mriqc.qc.anatomical.rpve`): the
  :abbr:`rPVe (residual partial voluming error)` of :abbr:`CSF (cerebrospinal fluid)`,
  :abbr:`GM (gray-matter)` and :abbr:`WM (white-matter)`. Lower values are better.

- **inu_\*** (*nipype interface to N4ITK*): summary statistics (max, min and median)
  of the :abbr:`INU (intensity non-uniformity)` field as extracted by the N4ITK algorithm
  [Tustison2010]_. Values closer to 1.0 are better.

- **summary_\*_\*** (:py:func:`~mriqc.qc.anatomical.summary_stats`):
  Mean, standard deviation, 5% percentile and 95% percentile of the distribution
  of background, :abbr:`CSF (cerebrospinal fluid)`, :abbr:`GM (gray-matter)` and
  :abbr:`WM (white-matter)`.

Most of these :abbr:`IQMs (image quality metrics)` are migrated or derivated from 
[QAP-measures]_.


Functional images (:abbr:`fMRI (functional MRI)`)
-------------------------------------------------

The :abbr:`IQMs (image quality metrics)` computed are as follows:

#. :py:func:`~mriqc.qc.anatomical.efc` Entropy Focus Criterion
#. **fber** - Foreground to Background Energy Ratio
#. **fwhm** - Full-width half maximum smoothness of the voxels averaged
   across the three coordinate axes, and also for each axis [x,y,x]
#. **ghost\_x** - Ghost to Signal Ratio
#. **snr** - Signal to Noise Ratio
#. **dvars** - Spatial standard deviation of the voxelwise temporal
   derivates
#. **gcor** - Global Correlation
#. **mean\_fd** - Mean Fractional Displacement
#. **num\_fd** - Number of volumes with :abbr:`FD (frame displacement)` greater than 0.2mm
#. **perc\_fd** - Percent of volumes with :abbr:`FD (frame displacement)` greater than 0.2mm
#. **outlier** - Mean fraction of outliers per fMRI volume
#. **quality** - Median Distance Index
#. **summary\_{mean, stdv, p05, p95}\_\*** - Mean, standard deviation, 5% percentile and 95% percentile of the distribution of background and foreground.


References
----------

  .. [Atkinson1997] Atkinson et al., *Automatic correction of motion artifacts
    in magnetic resonance images using an entropy
    focus criterion*, IEEE Trans Med Imag 16(6):903-910, 1997.
    doi:`10.1109/42.650886 <http://dx.doi.org/10.1109/42.650886>`_.

  .. [Dietrich2007] Dietrich et al., *Measurement of SNRs in MR images: influence
    of multichannel coils, parallel imaging and reconstruction filters*, JMRI 26(2):375--385.
    2007. doi:`10.1002/jmri.20969 <http://dx.doi.org/10.1002/jmri.20969>`_.

  .. [Friedman2008] Friedman, L et al., *Test--retest and between‐site reliability in a multicenter 
    fMRI study*. Hum Brain Mapp, 29(8):958--972, 2008. doi:`10.1002/hbm.20440
    <http://dx.doi.org/10.1002/hbm.20440>`_.

  .. [Ganzetti2016] Ganzetti et al., *Intensity inhomogeneity correction of structural MR images:
    a data-driven approach to define input algorithm parameters*. Front Neuroinform 10:10. 2016.
    doi:`10.3389/finf.201600010 <http://dx.doi.org/10.3389/finf.201600010>`_.

  .. [Giannelli2010] Giannelli et al., *Characterization of Nyquist ghost in
    EPI-fMRI acquisition sequences implemented on two clinical 1.5 T MR scanner
    systems: effect of readout bandwidth and echo spacing*. J App Clin Med Phy,
    11(4). 2010.
    doi:`10.1120/jacmp.v11i4.3237 <http://dx.doi.org/10.1120/jacmp.v11i4.3237>`_.

  .. [Jenkinson2002] Jenkinson et al., *Improved Optimisation for the Robust and
    Accurate Linear Registration and Motion Correction of Brain Images*.
    NeuroImage, 17(2), 825-841, 2002.
    doi:`10.1006/nimg.2002.1132 <http://dx.doi.org/10.1006/nimg.2002.1132>`_.

  .. [Kaufman1989] Kaufman et al., *Measuring signal-to-noise ratios in MR imaging*,\
    Radiology 173(1)265--267, 1989. doi:`10.1148/radiology.173.1.2781018
    <http://dx.doi.org/10.1148/radiology.173.1.2781018>`_

  .. [Magnota2006] Magnotta, VA., & Friedman, L., *Measurement of signal-to-noise
    and contrast-to-noise in the fBIRN multicenter imaging study*. 
    J Dig Imag 19(2):140-147, 2006. doi:`10.1007/s10278-006-0264-x
    <http://dx.doi.org/10.1007/s10278-006-0264-x>`_.

  .. [Mortamet2009] Mortamet B et al., *Automatic quality assessment in
    structural brain magnetic resonance imaging*, Mag Res Med 62(2):365-372,
    2009. doi:`10.1002/mrm.21992 <http://dx.doi.org/10.1002/mrm.21992>`_.

  .. [Nichols2013] Nichols, `Notes on Creating a Standardized Version of DVARS
      <http://www2.warwick.ac.uk/fac/sci/statistics/staff/academic-research/nichols/scripts/fsl/standardizeddvars.pdf>`_, 2013.

  .. [Power2012] Poweret al., *Spurious but systematic correlations in
    functional connectivity MRI networks arise from subject motion*,
    NeuroImage 59(3):2142-2154,
    2012, doi:`10.1016/j.neuroimage.2011.10.018
    <http://dx.doi.org/10.1016/j.neuroimage.2011.10.018>`_.

  .. [QAP] `The QAP project
    <https://github.com/oesteban/quality-assessment-protocol/blob/enh/SmartQCWorkflow/qap/temporal_qc.py#L16>`_.

  .. [Tustison2010] Tustison NJ et al., *N4ITK: improved N3 bias correction*, IEEE Trans Med Imag, 29(6):1310-20,
    2010. doi:`10.1109/TMI.2010.2046908 <http://dx.doi.org/10.1109/TMI.2010.2046908>`_

  .. [QAP-measures] `The Quality Assessment Protocols website: Taxonomy of QA Measures
    <http://preprocessed-connectomes-project.github.io/quality-assessment-protocol/#taxonomy-of-qa-measures>`_.