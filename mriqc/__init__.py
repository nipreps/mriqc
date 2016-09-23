#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
The mriqc package provides a series of :abbr:`NR (no-reference)`,
:abbr:`IQMs (image quality metrics)` to used in :abbr:`QAPs (quality
assessment protocols)` for :abbr:`MRI (magnetic resonance imaging)`.

"""
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from __future__ import unicode_literals
import logging

from .info import (
    __versionbase__,
    __versionrev__,
    __version__,
    __author__,
    __email__,
    __maintainer__,
    __copyright__,
    __credits__,
    __license__,
    __status__,
    __description__,
    __longdesc__
)

MRIQC_LOG = logging.getLogger('mriqc')
MRIQC_LOG.setLevel(logging.INFO)
