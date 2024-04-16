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
"""Manipulate Python warnings."""

import logging
import sys

from nipype import logging as nlogging

logging.addLevelName(26, 'BANNER')  # Add a new level for banners
logging.addLevelName(25, 'IMPORTANT')  # Add a new level between INFO and WARNING
logging.addLevelName(15, 'VERBOSE')  # Add a new level between INFO and DEBUG

LOGGER_FMT = (
    '%(asctime)s |{color} %(levelname)-8s {reset}|{color} %(name)-16s '
    '{reset}|{color} %(message)s{reset}'
)
DATE_FMT = '%Y-%m-%d %H:%M:%S'
CONSOLE_COLORS = {
    logging.DEBUG: '\x1b[38;20m',
    logging.INFO: '\x1b[34;20m',
    25: '\x1b[33;20m',
    logging.WARNING: '\x1b[93;20m',
    logging.ERROR: '\x1b[31;20m',
    logging.CRITICAL: '\x1b[31;1m',
    'reset': '\x1b[0m',
}


class _LogFormatter(logging.Formatter):
    """Customize the log format."""

    _colored = True

    def __init__(self, datefmt=None, colored=True, **kwargs):
        self._colored = colored
        super().__init__(
            datefmt=datefmt or DATE_FMT,
            fmt=LOGGER_FMT.format(
                color=CONSOLE_COLORS['reset'] if colored else '',
                reset=CONSOLE_COLORS['reset'] if colored else '',
            ),
        )

    def format(self, record):
        reset = CONSOLE_COLORS['reset'] if self._colored else ''
        self._style._fmt = (
            '%(message)s'
            if record.levelno == 26
            else LOGGER_FMT.format(
                color=CONSOLE_COLORS.get(
                    record.levelno,
                    CONSOLE_COLORS['reset'],
                )
                if self._colored
                else '',
                reset=reset,
            )
        )
        return super().format(record)


nlogging.getLogger('nipype')
_wlog = logging.getLogger('py.warnings')
_numexprlog = logging.getLogger('numexpr.utils')
_dataladlog = logging.getLogger('datalad')

for logger_name in logging.root.manager.loggerDict:
    logging.getLogger(logger_name).handlers.clear()

_root_logger = logging.getLogger()
# _root_logger.handlers.clear()
_handler = logging.StreamHandler(stream=sys.stdout)
_handler.setFormatter(_LogFormatter())
_root_logger.addHandler(_handler)

_wlog.addHandler(logging.NullHandler())
_numexprlog.addHandler(logging.NullHandler())
_dataladlog.addHandler(logging.NullHandler())

# def _warn(message, category=None, stacklevel=1, source=None):
#     """Redefine the warning function."""
#     if category is not None:
#         category = type(category).__name__
#         category = category.replace('type', 'WARNING')

#     logging.getLogger('py.warnings').debug(f"{category or 'WARNING'}: {message}")


# def _showwarning(message, category, filename, lineno, file=None, line=None):
#     _warn(message, category=category)


# warnings.warn = _warn
# warnings.showwarning = _showwarning

# warnings.filterwarnings("ignore", category=FutureWarning)
# warnings.filterwarnings("ignore", category=DeprecationWarning)
# warnings.filterwarnings("ignore", category=ResourceWarning)
# # cmp is not used by mriqc, so ignore nipype-generated warnings
# warnings.filterwarnings("ignore", "cmp not installed")
# warnings.filterwarnings(
#     "ignore", "This has not been fully tested. Please report any failures."
# )
# warnings.filterwarnings("ignore", "sklearn.externals.joblib is deprecated in 0.21")
# warnings.filterwarnings("ignore", "can't resolve package from __spec__ or __package__")
