mriqc: image quality metrics for quality assessment of MRI
==========================================================

|DOI| |Zenodo| |Package| |Pythons| |DevStatus| |License| |Documentation| |CircleCI|

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
   `CircleCI <https://circleci.com/gh/nipreps/mriqc>`_.

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
https://github.com/nipreps/mriqc/issues.

License information
-------------------
*MRIQC* adheres to the
`general licensing guidelines <https://www.nipreps.org/community/licensing/>`__
of the *NiPreps framework*.

*MRIQC* originally derives from, and hence is heavily influenced by, the
`PCP Quality Assessment Protocol
<http://preprocessed-connectomes-project.github.io/quality-assessment-protocol>`__.
Please check the ``NOTICE`` file for further information.

License
~~~~~~~
Copyright (c) 2021, the *NiPreps* Developers.

As of the 21.0.x pre-release and release series, *MRIQC* is
licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
`http://www.apache.org/licenses/LICENSE-2.0
<http://www.apache.org/licenses/LICENSE-2.0>`__.

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Acknowledgements
----------------
This work is steered and maintained by the `NiPreps Community <https://www.nipreps.org>`__.
The development of this resource was supported by
the Laura and John Arnold Foundation (RAP and KJG),
the NIBIB (R01EB020740, SSG; 1P41EB019936-01A1SSG, YOH),
the NIMH (RF1MH121867, RAP, OE; R24MH114705 and R24MH117179, RAP; 1RF1MH121885 SSG),
NINDS (U01NS103780, RAP), and NSF (CRCNS 1912266, YOH).
OE acknowledges financial support from the SNSF Ambizione project
“*Uncovering the interplay of structure, function, and dynamics of
brain connectivity using MRI*” (grant number
`PZ00P2_185872 <http://p3.snf.ch/Project-185872>`__).

.. topic:: **Thanks**

    * The QAP developers (C. Craddock, S. Giavasis, D. Clark, Z. Shezhad, and J.
      Pellman) for the initial base of code which MRIQC was forked from.
    * W Triplett and CA Moodie for their initial contributions with bugfixes and documentation, and
    * J Varada for his contributions on the source code.


.. |DOI| image:: https://img.shields.io/badge/doi-10.1371%2Fjournal.pone.0184661-blue.svg
   :target: https://doi.org/10.1371/journal.pone.0184661
.. |Zenodo| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.2630889.svg
   :target: https://doi.org/10.5281/zenodo.2630889
.. |Package| image:: https://img.shields.io/pypi/v/mriqc.svg
   :target: https://pypi.python.org/pypi/mriqc/
.. |Pythons| image:: https://img.shields.io/pypi/pyversions/mriqc.svg
   :target: https://pypi.python.org/pypi/mriqc/
.. |DevStatus| image:: https://img.shields.io/pypi/status/mriqc.svg
   :target: https://pypi.python.org/pypi/mriqc/
.. |License| image:: https://img.shields.io/pypi/l/mriqc.svg
   :target: https://pypi.python.org/pypi/mriqc/
.. |Documentation| image:: https://readthedocs.org/projects/mriqc/badge/?version=latest
   :target: http://mriqc.readthedocs.io/en/latest/?badge=latest
.. |CircleCI| image:: https://circleci.com/gh/nipreps/mriqc/tree/master.svg?style=shield
   :target: https://circleci.com/gh/nipreps/mriqc/tree/master
