.. _petqc:

PETQC workflows
***************

MRIQC includes experimental support for positron emission tomography (PET) quality
control. When PET data are present in a BIDS dataset, the ``pet`` workflow is
initialized automatically and generates PET-specific image quality metrics.

Running PETQC
-------------

Executing PETQC does not require additional command line options beyond the
standard *MRIQC* interface. Simply run::

  mriqc <bids_root> <output_dir> participant

MRIQC will detect ``pet`` images and process them along with other modalities.

Interpreting PETQC outputs
--------------------------

Individual HTML reports for each PET run are written to
``<output_dir>/reports``. Summary metrics are collated in
``<output_dir>/pet.csv`` and a group report is produced when the ``group`` level
is executed. These outputs mirror those of the MRI workflows and can be used to
identify artifacts or outlier scans.
