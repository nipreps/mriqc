#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# @Author: oesteban
# @Date:   2016-01-05 11:29:40
# @Email:  code@oscaresteban.es
# @Last modified by:   oesteban
# @Last Modified time: 2016-11-10 17:14:23
"""
Anatomical tests
"""
# from __future__ import division, print_function, absolute_import, unicode_literals
# import os.path as op
# import numpy as np
# import pytest
# from scipy.stats import rice
# from mriqc.interfaces.anatomical import artifact_mask

# @pytest.mark.parametrize("sigma", [0.02, 0.03, 0.05, 0.08, 0.12, 0.15, 0.2, 0.4, 0.5])
# def test_qi1(sigma):
#     size = (50, 50, 50)
#     test_data = np.ones(size)
#     wmdata = np.zeros(size)
#     bgdata = np.ones(size)
#     wmdata[22:24, 22:24, 22:24] = 1
#     wm_mean = 100
#     test_data[wmdata > 0] = wm_mean
#     test_data += rice.rvs(0.77, scale=sigma*wm_mean, size=test_data.shape)
#     artmask = artifact_mask(test_data, bgdata, bgdata, zscore=2.)
#     qi1 = artmask.sum() / bgdata.sum()
#     assert qi1 > .0 and qi1 < 0.002
