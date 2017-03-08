
.. _bold:


BOLD images
-----------

The :abbr:`IQMs (image quality metrics)` computed are as follows:

#. :py:func:`~mriqc.qc.anatomical.efc` Entropy Focus Criterion
#. **fber** - Foreground to Background Energy Ratio
#. **fwhm** - Full-width half maximum smoothness of the voxels averaged
   across the three coordinate axes, and also for each axis [x,y,x]
#. **ghost\_x** - Ghost to Signal Ratio
#. **snr** - Signal to Noise Ratio
#. **dvars** - Spatial standard deviation of the voxelwise temporal
   derivatives (calculated after motion correction)
#. **gcor** - Global Correlation
#. **mean\_fd** - Mean Framewise Displacement (as in Power et al. 2012)
#. **num\_fd** - Number of volumes with :abbr:`FD (frame displacement)` greater than 0.2mm
#. **perc\_fd** - Percent of volumes with :abbr:`FD (frame displacement)` greater than 0.2mm
#. **outlier** - Mean fraction of outliers per fMRI volume
#. **quality** - Median Distance Index
#. **summary\_{mean, stdv, p05, p95}\_\*** - Mean, standard deviation, 5% percentile and 95% percentile of the distribution of background and foreground.