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

__version__ = '0.0.2rc2'
