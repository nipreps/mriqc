# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""MRIQC."""

from ._version import get_versions

__version__ = get_versions()["version"]
del get_versions

__copyright__ = (
    "Copyright 2020, Center for Reproducible Neuroscience, Stanford University"
)
__credits__ = "Oscar Esteban"
__download__ = f"https://github.com/poldracklab/mriqc/archive/{__version__}.tar.gz"
__all__ = ["__version__", "__copyright__", "__credits__", "__download__"]
