mriqc: image quality metrics for quality assessment of MRI
==========================================================

MRIQC is developed by `the Poldrack Lab at Stanford University
<https://poldracklab.stanford.edu>`_ for use at the `Center for Reproducible
Neuroscience (CRN) <http://reproducibility.stanford.edu>`_, as well as
for open-source software distribution.

.. image:: http://bids.neuroimaging.io/openneuro_badge.svg
  :target: https://openneuro.org
  :alt: Available in OpenNeuro!

.. image:: https://circleci.com/gh/poldracklab/mriqc/tree/master.svg?style=svg
  :target: https://circleci.com/gh/poldracklab/mriqc/tree/master

.. image:: https://readthedocs.org/projects/mriqc/badge/?version=latest
  :target: http://mriqc.readthedocs.io/en/latest/?badge=latest
  :alt: Documentation Status

.. image:: https://api.codacy.com/project/badge/grade/fbb12f660141411a89ba1ae5bf873717
  :target: https://www.codacy.com/app/code_3/mriqc

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

#. **Modularity and integrability**: MRIQC implements a
   `nipype <http://nipype.readthedocs.io>`_ workflow to integrate modular 
   sub-workflows that rely upon third party software toolboxes such as 
   FSL, ANTs and AFNI.

#. **Minimal preprocessing**: the MRIQC workflows should be as minimal
   as possible to estimate the IQMs on the original data or their minimally
   processed derivatives.

#. **Interoperability and standards**: MRIQC follows the the `brain imaging data structure
   (BIDS) <http://bids.neuroimaging.io>`_, and it adopts the `BIDS-App
   <http://bids-apps.neuroimaging.io>`_ standard.
   
#. **Reliability and robustness**: the software undergoes frequent vetting sprints
   by testing its robustness against data variability (acquisition parameters,
   physiological differences, etc.) using images from `OpenfMRI <https://openfmri.org>`_.
   Its reliability is permanently checked and maintained with 
   `CircleCI <https://circleci.com/gh/poldracklab/mriqc>`_.


MRIQC is part of the MRI image analysis and reproducibility platform offered by
the CRN. This pipeline derives from, and is heavily influenced by, the
`PCP Quality Assessment Protocol <http://preprocessed-connectomes-project.github.io/quality-assessment-protocol>`_.

Citation
--------

.. topic:: **When using MRIQC, please include the following citation:**

    Esteban O, Birman D, Schaer M, Koyejo OO, Poldrack RA, Gorgolewski KJ;
    *MRIQC: Advancing the Automatic Prediction of Image Quality in MRI from Unseen Sites*;
    PLOS ONE 12(9):e0184661; doi:`10.1371/journal.pone.0184661 <https://doi.org/10.1371/journal.pone.0184661>`_.


Support and communication
-------------------------

The documentation of this project is found here: http://mriqc.readthedocs.io/.

Users can get help using the `mriqc-users google group <https://groups.google.com/forum/#!forum/mriqc-users>`_.

All bugs, concerns and enhancement requests for this software can be submitted here:
https://github.com/poldracklab/mriqc/issues.


Authors
-------

Oscar Esteban, Krzysztof F. Gorgolewski.
Poldrack Lab, Psychology Department, Stanford University,
and Center for Reproducible Neuroscience, Stanford University.

.. topic:: **Thanks**

    * The QAP developers (C. Craddock, S. Giavasis, D. Clark, Z. Shezhad, and J.
      Pellman) for the initial base of code which MRIQC was forked from.
    * W Triplett and CA Moodie for their initial contributions with bugfixes and documentation, and
    * J Varada for his contributions on the source code.


License information
-------------------

We use the 3-clause BSD license; the full license is in the file ``LICENSE`` in
the ``mriqc`` distribution.

All trademarks referenced herein are property of their respective
holders.

Copyright (c) 2015-2017, the mriqc developers and the CRN.
All rights reserved.
