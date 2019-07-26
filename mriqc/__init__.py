# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
The mriqc package provides a series of :abbr:`NR (no-reference)`,
:abbr:`IQMs (image quality metrics)` to used in :abbr:`QAPs (quality
assessment protocols)` for :abbr:`MRI (magnetic resonance imaging)`.

"""
import sys
import logging

from .__about__ import (
    __copyright__,
    __credits__,
    __version__,
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

__all__ = [
    '__copyright__',
    '__credits__',
    '__version__',
    'MRIQC_LOG',
]
