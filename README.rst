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

MRIQC extracts no-reference IQMs (image quality metrics) from
structural (T1w and T2w) and functional MRI (magnetic resonance imaging)
data.

MRIQC is an open-source project, developed under the following
software engineering principles:

1. Modularity and integrability: MRIQC implements a
nipype workflow to integrate modular sub-workflows that rely upon third
party software toolboxes such as FSL, ANTs and AFNI.
2. Minimal preprocessing: the workflow described before should be as minimal
as possible to estimate the IQMs on the original data or their minimally processed
derivatives.
3. Interoperability and standards: MRIQC follows the the brain imaging data structure
(BIDS), and it adopts the BIDS-App standard.
4. Reliability and robustness: the software undergoes frequent vetting sprints
by testing its robustness against data variability (acquisition parameters,
physiological differences, etc.) using images from the OpenfMRI resource.
Reliability is checked and maintained with the use of a continuous
integration service.


MRIQC is part of the MRI image analysis and reproducibility platform offered by
the CRN. This pipeline derives from, and is heavily influenced by, the
`PCP Quality Assessment Protocol <http://preprocessed-connectomes-project.github.io/quality-assessment-protocol>`_.


Support and communication
-------------------------

The documentation of this project is found here: http://mriqc.readthedocs.org/.

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

Copyright (c) 2015-2017, the mriqc developers and the CRN.
All rights reserved.
