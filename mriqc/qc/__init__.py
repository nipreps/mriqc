#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Some no-reference :abbr:`IQMs (image quality metrics)` are extracted in the
final stage of all processing workflows run by MRIQC.
A no-reference :abbr:`IQM (image quality metric)` is a measurement of some aspect
of the actual image which cannot be compared to a reference value for the metric
since there is no ground-truth about what this number should be.
All the computed :abbr:`IQMs (image quality metrics)` corresponding to
an image are saved in a `JSON file <iqm_json>`_ under the ``<output-dir>/derivatives/``
folder.

The IQMs can be grouped in four broad categories, providing a vector of 56
features per anatomical image. Some measures characterize the impact of noise and/or
evaluate the fitness of a noise model. A second family of measures use information
theory and prescribed masks to evaluate the spatial distribution of information. A third
family of measures look for the presence and impact of particular artifacts. Specifically,
the INU artifact, and the signal leakage due to rapid motion (e.g. eyes motion or blood
vessel pulsation) are identified. Finally, some measures that do not fit within the
previous categories characterize the statistical properties of tissue distributions, volume
overlap of tissues with respect to the volumes projected from MNI space, the
sharpness/blurriness of the images, etc.

.. note ::

  Most of the :abbr:`IQMs (image quality metrics)` in this module are adapted, derived or
  reproduced from the :abbr:`QAP (quality assessment protocols)` project [QAP]_.
  We particularly thank Steve Giavasis (`@sgiavasis <http://github.com/sgiavasis>`_) and
  Krishna Somandepali for their original implementations of the code in this module that
  we took from the [QAP]_.
  The [QAP]_ has a very good description of the :abbr:`IQMs (image quality metrics)`
  in [QAP-measures]_.


.. toctree::
    :maxdepth: 3

    iqms/t1w
    iqms/bold



.. topic:: References

  .. [QAP] `The QAP project
    <https://github.com/oesteban/quality-assessment-protocol/blob/enh/SmartQCWorkflow/qap/temporal_qc.py#L16>`_.

  .. [QAP-measures] `The Quality Assessment Protocols website: Taxonomy of QA Measures
    <http://preprocessed-connectomes-project.github.io/quality-assessment-protocol/#taxonomy-of-qa-measures>`_.


"""
