#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
""" mriqc nipype interfaces """
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from __future__ import unicode_literals

from mriqc.interfaces.anatomical import StructuralQC, ArtifactMask, ComputeQI2
from mriqc.interfaces.functional import FunctionalQC, Spikes
from mriqc.interfaces.bids import ReadSidecarJSON, IQMFileSink
from mriqc.interfaces.viz import PlotMosaic, PlotContours, PlotSpikes
from mriqc.interfaces.common import ConformImage
