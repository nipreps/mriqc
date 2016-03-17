MRIQC measures
==============

Structural images (:abbr:`sMRI (structural MRI)`)
-------------------------------------------------

The :abbr:`IQMs (image quality metrics)` computed are as follows:

#. **bg\_size** - Background mask size
#. **fg\_size** - Foreground mask size
#. **bg\_mean** - Mean intensity of the background mask
#. **fg\_mean** - Mean intensity of the foreground mask
#. **bg\_std** - Standard deviation of the background mask
#. **fg\_std** - Standard deviation of the foreground mask
#. **csf\_size** - Cerebrospinal fluid mask size
#. **gm\_size** - Grey matter mask size
#. **wm\_size** - White matter mask size
#. **csf\_mean** - Mean intensity of the CSF mask
#. **gm\_mean** - Mean intensity of the grey matter mask
#. **wm\_mean** - Mean intensity of the white matter mask
#. **csf\_std** - Standard deviation of the CSF mask
#. **gm\_std** - Standard deviation of the grey matter mask
#. **wm\_std** - Standard deviation of the white matter mask
#. **cnr** - Contrast to Noise Ratio
#. **efc** - Entropy Focus Criterion
#. **fber** - Foreground to Background Energy Ratio
#. **fwhm** - Full-width half maximum smoothness of the voxels averaged
   across the three coordinate axes, and also for each axis [x,y,x]
#. **qi1** - Artifact Detection
#. **snr** - Signal to Noise Ratio

All these metrics are described in more detail in the `Taxonomy of QA Measures
section <http://preprocessed-connectomes-project.github.io/quality-assessment-protocol/#taxonomy-of-qa-measures>`_
of the QAP documentation. Please refer to the QAP website for
descriptions of these metrics.


Functional images (:abbr:`fMRI (functional MRI)`)
-------------------------------------------------

The Spatial Metrics computed on the Functional Scan
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The metrics displayed in the Summary Report were computed using the

::

    qap_functional_spatial.py 

workflow and have been displayed as violin plots. The stars in these
plots denote where the score for this particular scan falls in the
distribution of all scores for scans that were included as inputs to
this workflow.

The metrics computed are as follows:

#. bg\_size - Background mask size
#. fg\_size - Foreground mask size
#. bg\_mean - Mean intensity of the background mask
#. fg\_mean - Mean intensity of the foreground mask
#. bg\_std - Standard deviation of the background mask
#. fg\_std - Standard deviation of the foreground mask
#. efc - Entropy Focus Criterion
#. fber - Foreground to Background Energy Ratio
#. fwhm - Full-width half maximum smoothness of the voxels averaged
   across the three coordinate axes, and also for each axis [x,y,x]
#. ghost\_x - Ghost to Signal Ratio
#. snr - Signal to Noise Ratio

All metrics are described in more detail in the `Taxonomy of QA Measures
section <http://preprocessed-connectomes-project.github.io/quality-assessment-protocol/#taxonomy-of-qa-measures>`__
of the QAP documentation. Please refer to the QAP website for
descriptions of these metrics.

The Temporal Metrics computed on the Functional Scan
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The metrics displayed in the Summary Report were computed using the
``qap_functional_temporal.py`` workflow and have been displayed as
violin plots. Eg:

::

    QC measures (subject sub-01_session_1)

The stars in these plots denote where the score for this particular scan
falls in the distribution of all scores for scans that were included as
inputs to the the functional-temporal workflow.

The metrics computed are as follows:

#. dvars - Spatial standard deviation of the voxelwise temporal
   derivates
#. gcor - Global Correlation
#. mean\_fd - Mean Fractional Displacement
#. num\_fd - Number of volumes with :abbr:`FD (frame displacement)` greater than 0.2mm
#. perc\_fd - Percent of volumes with :abbr:`FD (frame displacement)` greater than 0.2mm
#. outlier - Mean fraction of outliers per fMRI volume
#. quality - Median Distance Index

All metrics are described in more detail in the `Taxonomy of QA Measures
section <http://preprocessed-connectomes-project.github.io/quality-assessment-protocol/#taxonomy-of-qa-measures>`__
of the QAP documentation. Please refer to the QAP website for
descriptions of these metrics.