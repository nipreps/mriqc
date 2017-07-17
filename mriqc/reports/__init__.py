#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""

In order to ease the screening process of individual images, MRIQC
generates individual reports with mosaic views of a number of cutting planes and
supporting information (for example, segmentation contours). The most straightforward
use-case is the visualization of those images flagged as low-quality by the classifier.

After the extraction of :abbr:`IQMs (image quality metrics)` in all the images of our sample,
a group report is generated. The group report shows a scatter plot for each of the
:abbr:`IQMs (image quality metrics)`, so it is particularly easy to identify the cases that
are outliers for each metric. The plots are interactive, such that clicking on any particular
sample opens the corresponding individual report of that case. Examples of group and individual
reports for the ABIDE dataset are available online at `mriqc.org <http://mriqc.org>`_.

.. toctree::
    :maxdepth: 3

    reports/group
    reports/smri
    reports/bold

mriqc.reports package
=====================

Submodules
----------

.. automodule:: mriqc.reports.group
    :members:
    :undoc-members:
    :show-inheritance:

.. automodule:: mriqc.reports.individual
    :members:
    :undoc-members:
    :show-inheritance:

.. automodule:: mriqc.reports.utils
    :members:
    :undoc-members:
    :show-inheritance:

"""
from __future__ import print_function, division, absolute_import, unicode_literals
from .individual import individual_html
from .group import gen_html as group_html
