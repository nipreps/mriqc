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
