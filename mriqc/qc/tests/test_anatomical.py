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
"""
Anatomical tests
"""
from builtins import object
from shutil import rmtree
from tempfile import mkdtemp

import numpy as np
import pytest
from scipy.stats import rice

# from numpy.testing import allclose
from ..anatomical import art_qi2


class GroundTruth(object):
    def get_data(self, sigma, noise="normal"):
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

        if noise == "rice":
            test_data += rice.rvs(0.77, scale=sigma * wm_mean, size=test_data.shape)
        elif noise == "rayleigh":
            test_data += np.random.rayleigh(scale=sigma * wm_mean, size=test_data.shape)
        else:
            test_data += np.random.normal(
                0.0, scale=sigma * wm_mean, size=test_data.shape
            )

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
    assert value > 0.0 and value < 0.04
