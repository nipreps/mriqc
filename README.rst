mriqc
=====

The mriqc package provides a series of NR (no-reference),
IQMs (image quality metrics) to used in QAPs (quality
assessment protocols) for MRI (magnetic resonance imaging).


Dependencies
------------

Make sure you have FSL, N4ITK (released with ANTs), and AFNI installed, and the binaries are available in
the system's ``$PATH``.


Installation
------------

Just issue ::

    pip install mriqc


Example command line ::

    mriqc -i ~/Data/bids_dataset -o out/ -w work/
