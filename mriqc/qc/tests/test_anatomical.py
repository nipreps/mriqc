#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# @Author: oesteban
# @Date:   2016-01-05 11:29:40
# @Email:  code@oscaresteban.es
# @Last modified by:   oesteban
# @Last Modified time: 2018-08-21 14:52:07
"""
Anatomical tests
"""
from __future__ import division, print_function, absolute_import, unicode_literals
from tempfile import mkdtemp
from shutil import rmtree
import numpy as np
import pytest
from scipy.stats import rice
from builtins import object
# from numpy.testing import allclose
from ..anatomical import art_qi2


class GroundTruth(object):

    def get_data(self, sigma, noise='normal'):
        """Generates noisy 3d data"""
        size = (50, 50, 50)
        test_data = np.ones(size)
        wmdata = np.zeros(size)
        bgdata = np.zeros(size)
        bgdata[:, :25, :] = 1
        wmdata[bgdata == 0] = 1

        bg_mean = 0
        wm_mean = 600
        test_data[bgdata > 0] = bg_mean
        test_data[wmdata > 0] = wm_mean

        if noise == 'rice':
            test_data += rice.rvs(0.77, scale=sigma * wm_mean, size=test_data.shape)
        elif noise == 'rayleigh':
            test_data += np.random.rayleigh(scale=sigma * wm_mean, size=test_data.shape)
        else:
            test_data += np.random.normal(0., scale=sigma * wm_mean, size=test_data.shape)

        return test_data, wmdata, bgdata


@pytest.fixture
def gtruth():
    return GroundTruth()


# @pytest.mark.parametrize("sigma", [0.01, 0.03, 0.05, 0.08, 0.12, 0.15, 0.20])
# def test_cjv(sigma, rtol=0.1):
#     size = (50, 50)
#     test_data = np.ones(size)
#     wmdata = np.zeros(size)
#     gmdata = np.zeros(size)
#     gmdata[:, :25] = 1
#     wmdata[gmdata == 0] = 1

#     gm_mean = 200
#     wm_mean = 600
#     test_data[gmdata > 0] = gm_mean
#     test_data[wmdata > 0] = wm_mean

#     test_data[wmdata > .5] += np.random.normal(
#         0.0, scale=sigma*wm_mean, size=test_data[wmdata > .5].shape)
#     test_data[gmdata > .5] += np.random.normal(
#         0.0, scale=sigma*gm_mean, size=test_data[gmdata > .5].shape)

#     exp_cjv = sigma * (wm_mean + gm_mean) / (wm_mean - gm_mean)

#     assert np.isclose(cjv(test_data, wmmask=wmdata, gmmask=gmdata), exp_cjv, rtol=rtol)


# @pytest.mark.parametrize("sigma", [0.02, 0.03, 0.05, 0.08, 0.12, 0.15, 0.2, 0.4, 0.5])
# @pytest.mark.parametrize("noise", ['normal', 'rice'])
# def test_snr(gtruth, sigma, noise):
#     data, wmdata, _ = gtruth.get_data(sigma, noise)
#     assert abs(snr(data, wmdata) - (1/sigma)) < 20


# @pytest.mark.parametrize("sigma", [0.02, 0.03, 0.05, 0.08, 0.12, 0.15, 0.2, 0.4, 0.5])
# @pytest.mark.parametrize("noise", ['rice', 'rayleigh'])
# def test_snr_dietrich(gtruth, sigma, noise):
#     data, wmdata, bgdata = gtruth.get_data(sigma, noise)
#     assert abs(snr_dietrich(data, wmdata, bgdata) - (1/sigma)) < 10


@pytest.mark.parametrize("sigma", [0.02, 0.03, 0.05, 0.08, 0.12, 0.15, 0.2, 0.4, 0.5])
def test_qi2(gtruth, sigma):
    tmpdir = mkdtemp()
    data, _, bgdata = gtruth.get_data(sigma, rice)
    value, _ = art_qi2(data, bgdata, save_plot=False)
    rmtree(tmpdir)
    assert value > .0 and value < 0.004
