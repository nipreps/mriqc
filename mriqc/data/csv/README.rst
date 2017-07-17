
README
======

This folder is organized as follows:

- ``archive`` contains old IQMs and labels for the alpha
  version of MRIQC (<0.9.1)
- ``manual_qc`` contains the output of manual labelings done
  on ABIDE and DS030 by Dan Birman (DB) and Marie Schaer (MS).
  These csv files are the outputs of the labeler script.
- ``x_<dataset>_<version>.csv`` are the IQMs extracted on the
  dataset ``<dataset>`` with version ``<version>`` of MRIQC.
- ``y_<dataset>.csv`` are ratings for dataset ``<dataset>``
- ``raters_merge.py`` is a script to merge in ratings from
  several raters.
- ``scan_parameters.tsv`` is a table with the scanning parameters
  of each sample.