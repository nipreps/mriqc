#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
The mriqc package provides a series of :abbr:`NR (no-reference)`,
:abbr:`IQMs (image quality metrics)` to used in :abbr:`QAPs (quality
assessment protocols)` for :abbr:`MRI (magnetic resonance imaging)`.

"""
from __future__ import print_function, division, absolute_import, unicode_literals
import sys
import logging

from .__about__ import (
    __version__,
    __author__,
    __email__,
    __maintainer__,
    __copyright__,
    __credits__,
    __license__,
    __status__,
    __description__,
    __longdesc__,
    __url__,
    __download__,
)

LOG_FORMAT = '%(asctime)s %(name)s:%(levelname)s %(message)s'
MRIQC_LOG = logging.getLogger()
logging.basicConfig(
    stream=sys.stdout,
    level=logging.WARNING,
    format=LOG_FORMAT,
)

# Add two levels of verbosity to info
logging.addLevelName(19, 'INFO')
logging.addLevelName(18, 'INFO')

DEFAULTS = {
    'ants_nthreads': 6,
    'float32': False
}
