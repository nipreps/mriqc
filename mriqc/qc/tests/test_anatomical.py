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
from __future__ import division, print_function, absolute_import, unicode_literals
import os.path as op
import numpy as np
import nibabel as nb
import pytest
from scipy.stats import rice
from niworkflows.data import get_brainweb_1mm_normal
from mriqc.qc.anatomical import snr, snr_dietrich, cjv, art_qi2

# from numpy.testing import allclose


class GroundTruth:
    def __init__(self):
        self.data = op.join(get_brainweb_1mm_normal(), 'sub-normal01')
        self.wmmask = op.join(get_brainweb_1mm_normal(), 'derivatives', 'volume_fraction_wht.nii.gz')
        self.wmdata = nb.load(self.wmmask).get_data().astype(np.float32)
        self.airmask = op.join(get_brainweb_1mm_normal(), 'derivatives', 'volume_fraction_bck.nii.gz')
        self.airdata = nb.load(self.airmask).get_data().astype(np.float32)

        self.ses = 'ses-pn0rf00'
        self.im_file = op.join(self.data, self.ses, 'anat', 'sub-normal01_%s_T1w.nii.gz' % self.ses)
        self.imdata = nb.load(self.im_file).get_data()
        self.fg_mean = np.median(self.imdata[self.wmdata > .99])

    def get_data(self, sigma, noise):
        if noise == 'normal':
            ndata = np.random.normal(0.0, scale=sigma*self.fg_mean, size=self.imdata.shape)
        elif noise == 'rayleigh':
            ndata = np.random.rayleigh(scale=sigma*self.fg_mean, size=self.imdata.shape)

        test_data = self.imdata + ndata
        test_data[test_data < 0] = 0
        return test_data, self.wmdata, self.airdata


@pytest.fixture
def gtruth():
    return GroundTruth()


@pytest.mark.parametrize("sigma", [0.01, 0.03, 0.05, 0.08, 0.12, 0.15, 0.20])
def test_cjv(sigma, rtol=0.1):
    size = (50, 50)
    test_data = np.ones(size)
    wmdata = np.zeros(size)
    gmdata = np.zeros(size)
    gmdata[:, :25] = 1
    wmdata[gmdata == 0] = 1

    gm_mean = 200
    wm_mean = 600
    test_data[gmdata > 0] = gm_mean
    test_data[wmdata > 0] = wm_mean

    test_data[wmdata > .5] += np.random.normal(0.0, scale=sigma*wm_mean, size=test_data[wmdata > .5].shape)
    test_data[gmdata > .5] += np.random.normal(0.0, scale=sigma*gm_mean, size=test_data[gmdata > .5].shape)

    exp_cjv = sigma * (wm_mean + gm_mean) / (wm_mean - gm_mean)

    assert np.isclose(cjv(test_data, wmmask=wmdata, gmmask=gmdata), exp_cjv, rtol=rtol)


@pytest.mark.parametrize("sigma", [0.02, 0.03, 0.05, 0.08, 0.12, 0.15, 0.2, 0.4, 0.5])
@pytest.mark.parametrize("noise", ['normal', 'rayleigh'])
def test_snr(gtruth, sigma, noise):
    data = gtruth.get_data(sigma, noise)
    error = abs(snr(*data[:2]) - (1 / sigma)) * sigma
    assert  error < 6.0


@pytest.mark.parametrize("sigma", [0.02, 0.03, 0.05, 0.08, 0.12, 0.15, 0.2, 0.4, 0.5])
@pytest.mark.parametrize("noise", ['rice', 'rayleigh'])
def test_snr_dietrich(sigma, noise):
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
        test_data += rice.rvs(0.77, scale=sigma*wm_mean, size=test_data.shape)
    elif noise == 'rayleigh':
        test_data += np.random.rayleigh(scale=sigma*wm_mean, size=test_data.shape)

    assert abs(snr_dietrich(test_data, wmdata, bgdata) - (1/sigma)) < 10

@pytest.mark.parametrize('brainweb', [
    'sub-normal01_ses-pn3rf00_T1w.nii.gz',
    'sub-normal01_ses-pn5rf00_T1w.nii.gz',
    'sub-normal01_ses-pn9rf00_T1w.nii.gz'])
def test_artifacts(brainweb):
    data = op.join(get_brainweb_1mm_normal(), 'sub-normal01')
    airdata = nb.load(op.join(get_brainweb_1mm_normal(), 'derivatives',
                              'volume_fraction_bck.nii.gz')).get_data()

    fname = op.join(data, brainweb.split('_')[1], 'anat', brainweb)
    imdata = nb.load(fname).get_data().astype(np.float32)

    value, _ = art_qi2(imdata[::4,::4,::4], airdata[::4,::4,::4])
    assert value > .0 and value <= 1
