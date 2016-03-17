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

Just issue:

::

    pip install mriqc

Example command line:
---------------------

::

    mriqc -i ~/Data/bids_dataset -o out/ -w work/


"""

__version__ = '0.0.3a4'
__author__ = 'Oscar Esteban'
__email__ = 'code@oscaresteban.es'
__maintainer__ = 'Oscar Esteban'
__copyright__ = ('Copyright 2016, Center for Reproducible Neuroscience, '
                 'Stanford University')
__credits__ = 'Oscar Esteban'
__license__ = '3-clause BSD'
__status__ = 'Prototype'
__description__ = 'NR-IQMs (no-reference Image Quality Metrics) for MRI'
