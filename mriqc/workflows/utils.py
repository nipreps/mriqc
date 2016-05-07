#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# @Author: oesteban
# @Date:   2016-01-05 17:15:12
# @Email:  code@oscaresteban.es
# @Last modified by:   oesteban
# @Last Modified time: 2016-05-04 15:15:14
"""Helper functions for the workflows"""

def fmri_getidx(in_file, start_idx, stop_idx):
    """Heuristics to set the start and stop indices of fMRI series"""
    from nibabel import load
    from nipype.interfaces.base import isdefined
    nvols = load(in_file).shape[3]
    max_idx = nvols - 1

    if start_idx is None or not isdefined(start_idx) or start_idx < 0 or start_idx > max_idx:
        start_idx = 0

    if (stop_idx is None or not isdefined(stop_idx) or stop_idx < start_idx or
            stop_idx > max_idx):
        stop_idx = max_idx
    return start_idx, stop_idx

def fwhm_dict(fwhm):
    """Convert a list of FWHM into a dictionary"""
    fwhm = [float(f) for f in fwhm]
    return {'x': fwhm[0], 'y': fwhm[1],
            'z': fwhm[2], 'avg': fwhm[3]}
