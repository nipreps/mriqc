# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""mriqc nipype interfaces """

from mriqc.interfaces.anatomical import (
    ArtifactMask,
    ComputeQI2,
    Harmonize,
    RotationMask,
    StructuralQC,
)
from mriqc.interfaces.bids import IQMFileSink
from mriqc.interfaces.common import ConformImage, EnsureSize
from mriqc.interfaces.functional import FunctionalQC, Spikes
from mriqc.interfaces.viz import PlotContours, PlotMosaic, PlotSpikes
from mriqc.interfaces.webapi import UploadIQMs

__all__ = [
    "ArtifactMask",
    "ComputeQI2",
    "ConformImage",
    "EnsureSize",
    "FunctionalQC",
    "Harmonize",
    "IQMFileSink",
    "PlotContours",
    "PlotMosaic",
    "PlotSpikes",
    "RotationMask",
    "Spikes",
    "StructuralQC",
    "UploadIQMs",
]
