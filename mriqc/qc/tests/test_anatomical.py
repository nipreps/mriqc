#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# @Author: oesteban
# @Date:   2016-01-05 11:29:40
# @Email:  code@oscaresteban.es
# @Last modified by:   oesteban
# @Last Modified time: 2016-10-04 12:21:22
"""
Anatomical tests
"""
from __future__ import division, print_function, absolute_import, unicode_literals
import os.path as op
import nibabel as nb
import pytest

from niworkflows.data import get_brainweb_1mm_normal
from mriqc.qc.anatomical import snr, snr_dietrich, cjv, art_qi2
from mriqc.interfaces.anatomical import artifact_mask
import numpy as np
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


@pytest.mark.parametrize("sigma,exp_cjv", [
    (0.01, 0.45419429982401677),
    (0.03, 0.5114948933353829),
    (0.05, 0.6077553259357966),
    (0.08, 0.7945079788409393),
    (0.12, 1.0781050744254561),
    (0.15, 1.3013580100905036),
    (0.2, 1.6898180485107017)
])
def test_cjv(sigma, exp_cjv, rtol=0.1):
    data = op.join(get_brainweb_1mm_normal(), 'sub-normal01')

    wmmask = op.join(get_brainweb_1mm_normal(), 'derivatives', 'volume_fraction_wht.nii.gz')
    wmdata = nb.load(wmmask).get_data().astype(np.float32)
    gmmask = op.join(get_brainweb_1mm_normal(), 'derivatives', 'volume_fraction_gry.nii.gz')
    gmdata = nb.load(gmmask).get_data().astype(np.float32)

    ses = 'ses-pn0rf00'
    im_file = op.join(data, ses, 'anat', 'sub-normal01_%s_T1w.nii.gz' % ses)
    imdata = nb.load(im_file).get_data()

    fg_mean = np.mean(imdata[wmdata > .95])
    test_data = imdata + np.random.normal(0.0, scale=sigma*fg_mean, size=imdata.shape)
    test_data[test_data < 0] = 0
    assert abs(cjv(test_data, wmmask=wmdata, gmmask=gmdata) - exp_cjv) < rtol


@pytest.mark.parametrize("sigma", [0.02, 0.03, 0.05, 0.08, 0.12, 0.15, 0.2, 0.4, 0.5])
@pytest.mark.parametrize("noise", ['normal', 'rayleigh'])
def test_snr(gtruth, sigma, noise):
    data = gtruth.get_data(sigma, noise)
    error = abs(snr(*data[:2]) - (1 / sigma)) * sigma
    assert  error < 6.0


@pytest.mark.parametrize("sigma", [0.02, 0.03, 0.05, 0.08, 0.12, 0.15, 0.2, 0.4, 0.5])
@pytest.mark.parametrize("noise", ['normal', 'rayleigh'])
def test_snr_dietrich(gtruth, sigma, noise):
    data = gtruth.get_data(sigma, noise)
    error = abs(snr_dietrich(*data) - (1 / sigma)) * sigma
    assert  error < 6.0

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

    value = art_qi2(imdata[::4,::4,::4], airdata[::4,::4,::4], save_figure=False)
    assert value > .0 and value <= 1
