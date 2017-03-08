
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

.. topic:: References

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
