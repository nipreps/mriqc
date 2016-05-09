mriqc: image quality metrics for quality assessment of MRI
==========================================================

This pipeline is developed by `the Poldrack Lab at Stanford University
<https://poldracklab.stanford.edu>`_ for use at the `Center for Reproducible
Neuroscience (CRN) <http://reproducibility.stanford.edu>`_, as well as
for open-source software distribution.

.. image:: https://circleci.com/gh/poldracklab/mriqc/tree/master.svg?style=svg
  :target: https://circleci.com/gh/poldracklab/mriqc/tree/master

.. image:: https://readthedocs.org/projects/mriqc/badge/?version=latest
  :target: http://mriqc.readthedocs.io/en/latest/?badge=latest
  :alt: Documentation Status

.. image:: https://api.codacy.com/project/badge/grade/fbb12f660141411a89ba1ae5bf873717
  :target: https://www.codacy.com/app/code_3/mriqc

.. image:: https://coveralls.io/repos/github/poldracklab/mriqc/badge.svg?branch=master 
  :target: https://coveralls.io/github/poldracklab/mriqc?branch=master

.. image:: https://codecov.io/gh/poldracklab/mriqc/branch/master/graph/badge.svg
  :target: https://codecov.io/gh/poldracklab/mriqc

.. image:: https://img.shields.io/pypi/v/mriqc.svg
    :target: https://pypi.python.org/pypi/mriqc/
    :alt: Latest Version

.. image:: https://img.shields.io/pypi/dm/mriqc.svg
    :target: https://pypi.python.org/pypi/mriqc/
    :alt: Downloads

.. image:: https://img.shields.io/pypi/pyversions/mriqc.svg
    :target: https://pypi.python.org/pypi/mriqc/
    :alt: Supported Python versions

.. image:: https://img.shields.io/pypi/status/mriqc.svg
    :target: https://pypi.python.org/pypi/mriqc/
    :alt: Development Status

.. image:: https://img.shields.io/pypi/l/mriqc.svg
    :target: https://pypi.python.org/pypi/mriqc/
    :alt: License


About
-----

The package provides a series of image processing workflows to extract and
compute a series of NR (no-reference), IQMs (image quality metrics) to be 
used in QAPs (quality assessment protocols) for MRI (magnetic resonance imaging).

This open-source neuroimaging data processing tool is being developed as a
part of the MRI image analysis and reproducibility platform offered by
the CRN. This pipeline derives from, and is heavily influenced by, the
`PCP Quality Assessment Protocol <http://preprocessed-connectomes-project.github.io/quality-assessment-protocol>`_.

This tool extracts a series of IQMs from structural and functional MRI data.
It is also scheduled to add diffusion MRI to the target imaging families.


External Dependencies
---------------------

``mriqc`` is implemented using ``nipype``, but it requires some other neuroimaging
software tools:

- `FSL <http://fsl.fmrib.ox.ac.uk/fsl/fslwiki/>`_.
- The N4ITK bias correction tool released with `ANTs <http://stnava.github.io/ANTs/>`_.
- `AFNI <https://afni.nimh.nih.gov/>`_.

These tools must be installed and their binaries available in the 
system's ``$PATH``.


Installation
------------

The ``mriqc`` is packaged and available through the PyPi repository.
Therefore, the easiest way to install the tool is: ::

    pip install mriqc



Execution and the BIDS format
-----------------------------

The ``mriqc`` workflow takes as principal input the path of the dataset
that is to be processed.
The only requirement to the input dataset is that it has a valid `BIDS (Brain
Imaging Data Structure) <http://bids.neuroimaging.io/>`_ format.
This can be easily checked online using the 
`BIDS Validator <http://incf.github.io/bids-validator/>`_.

Example command line: ::

    mriqc -i ~/Data/bids_dataset -o out/ -w work/


Support and communication
-------------------------

The documentation of this project is found here: http://mriqc.readthedocs.org/en/latest/.

If you have a problem or would like to ask a question about how to use ``mriqc``,
please submit a question to NeuroStars.org with an ``mriqc`` tag.
NeuroStars.org is a platform similar to StackOverflow but dedicated to neuroinformatics.

All previous ``mriqc`` questions are available here:
http://neurostars.org/t/mriqc/

To participate in the ``mriqc`` development-related discussions please use the 
following mailing list: http://mail.python.org/mailman/listinfo/neuroimaging
Please add *[mriqc]* to the subject line when posting on the mailing list.


All bugs, concerns and enhancement requests for this software can be submitted here:
https://github.com/poldracklab/mriqc/issues.


Authors
-------

Oscar Esteban, Krzysztof F. Gorgolewski, Craig A. Moodie, William Triplett.
Poldrack Lab, Psychology Department, Stanford University,
and Center for Reproducible Neuroscience, Stanford University.


License information
-------------------

We use the 3-clause BSD license; the full license is in the file ``LICENSE`` in
the ``mriqc`` distribution.

All trademarks referenced herein are property of their respective
holders.

Copyright (c) 2015-2016, the mriqc developers and the CRN.
All rights reserved.
