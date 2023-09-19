# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# Copyright 2021 The NiPreps Developers <nipreps@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# We support and encourage derived works from this project, please read
# about our expectations at
#
#     https://www.nipreps.org/community/licensing/
#
"""
Generate individual and group reports.

In order to ease the screening process of individual images, MRIQC
generates individual reports with mosaic views of a number of cutting planes and
supporting information (for example, segmentation contours). The most straightforward
use-case is the visualization of those images flagged as low-quality by the classifier.

After the extraction of :abbr:`IQMs (image quality metrics)` in all the images of our sample,
a group report is generated. The group report shows a scatter plot for each of the
:abbr:`IQMs (image quality metrics)`, so it is particularly easy to identify the cases that
are outliers for each metric. The plots are interactive, such that clicking on any particular
sample opens the corresponding individual report of that case. Examples of group and individual
reports for the ABIDE dataset are available online at `mriqc.org <https://mriqc.readthedocs.io/>`_.

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

"""
