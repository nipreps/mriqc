MRIQC measures
==============

Structural images (:abbr:`sMRI (structural MRI)`)
-------------------------------------------------

The :abbr:`IQMs (image quality metrics)` computed are as follows:

+-------------------------+-------------------+-----------------------------------------+
| Identifier              | Reference         | Description                             |
+=========================+===================+=========================================+
| | **cnr**               |                   | :py:func:`~mriqc.qc.anatomical.cnr`     |
+-------------------------+-------------------+-----------------------------------------+
| | **efc**               |                   | Entropy Focus Criterion                 |
+-------------------------+-------------------+-----------------------------------------+
| | **fber**              |                   | Foreground to Background Energy Ratio   |
+-------------------------+-------------------+-----------------------------------------+
| | **fwhm**              |                   | Full-width half maximum smoothness of   |
|                         |                   | the voxels averaged across the three    |
|                         |                   | coordinate axes, and also for each axis |
+-------------------------+-------------------+-----------------------------------------+
| | **qi1**               |                   | Artifact Detection                      |
+-------------------------+-------------------+-----------------------------------------+
| | **snr**               |                   | Signal to Noise Ratio                   |
+-------------------------+-------------------+-----------------------------------------+
| | **bias\_\***          | [Tustison2010]_   | Summary measures of the bias field      |
|                         |                   | component estimated by *N4ITK*          |
+-------------------------+-------------------+-----------------------------------------+
| | **rpve\_\***          |                   | Residual Partial Volumes (rPVEs) of     |
|                         |                   | :abbr:`CSF (cerebrospinal fluid)`,      |
|                         |                   | :abbr:`GM (gray-matter)` and            |
|                         |                   | :abbr:`WM (white-matter)`.              |
+-------------------------+-------------------+-----------------------------------------+
| | **icvs\_\***          |                   | Intracranial volumes (ICVs) of          |
|                         |                   | :abbr:`CSF (cerebrospinal fluid)`,      |
|                         |                   | :abbr:`GM (gray-matter)` and            |
|                         |                   | :abbr:`WM (white-matter)`.              |
+-------------------------+-------------------+-----------------------------------------+
| | **summary\_mean\_\*** |                   | Mean, standard deviation, 5% percentile |
| | **summary\_stdv\_\*** |                   | and 95% percentile of the distribution  |
| | **summary\_p05\_\***  |                   | of background,                          |
| | **summary\_p95\_\***  |                   | :abbr:`CSF (cerebrospinal fluid)`,      |
|                         |                   | :abbr:`GM (gray-matter)` and            |
|                         |                   | :abbr:`WM (white-matter)`               |
+-------------------------+-------------------+-----------------------------------------+

Most of these :abbr:`IQMs (image quality metrics)` are migrated or derivated from [QAP]_.


Functional images (:abbr:`fMRI (functional MRI)`)
-------------------------------------------------

The :abbr:`IQMs (image quality metrics)` computed are as follows:

#. **efc** Entropy Focus Criterion
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

  .. [Tustison2010] Tustison NJ et al., *N4ITK: improved N3 bias correction*, IEEE Trans Med Imag, 29(6):1310-20, 2010. doi:`10.1109/TMI.2010.2046908 <http://dx.doi.org/10.1109/TMI.2010.2046908>`_
  .. [QAP] `The Quality Assessment Protocols website: Taxonomy of QA Measures
    <http://preprocessed-connectomes-project.github.io/quality-assessment-protocol/#taxonomy-of-qa-measures>`_.