#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
The mriqc package provides a series of :abbr:`NR (no-reference)`,
:abbr:`IQMs (image quality metrics)` to used in :abbr:`QAPs (quality
assessment protocols)` for :abbr:`MRI (magnetic resonance imaging)`.

Dependencies
------------

Make sure you have FSL and AFNI installed, and the binaries available in
the system's $PATH.

Installation
------------

The easiest path is using pypi ::

    pip install mriqc


Example
-------

::

    mriqc -i ~/Data/bids_dataset -o out/ -w work/


"""

__versionbase__ = '0.8.5'
__versionrev__ = 'a3'
__version__ = __versionbase__ + __versionrev__
__author__ = 'Oscar Esteban'
__email__ = 'code@oscaresteban.es'
__maintainer__ = 'Oscar Esteban'
__copyright__ = ('Copyright 2016, Center for Reproducible Neuroscience, '
                 'Stanford University')
__credits__ = 'Oscar Esteban'
__license__ = '3-clause BSD'
__status__ = 'Prototype'
__description__ = 'NR-IQMs (no-reference Image Quality Metrics) for MRI'
__longdesc__ = """
MRIQC provides a series of image processing workflows to extract and compute a series of
NR (no-reference), IQMs (image quality metrics) to be used in QAPs (quality assessment
protocols) for MRI (magnetic resonance imaging).
This open-source neuroimaging data processing tool is being developed as a part of the
MRI image analysis and reproducibility platform offered by the CRN. This pipeline derives
from, and is heavily influenced by, the PCP Quality Assessment Protocol.
This tool extracts a series of IQMs from structural and functional MRI data. It is also
scheduled to add diffusion MRI to the target imaging families.
"""
