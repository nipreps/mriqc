#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# @Author: oesteban
# @Date:   2016-01-05 11:29:40
# @Email:  code@oscaresteban.es
# @Last modified by:   oesteban
# @Last Modified time: 2016-04-08 10:40:08
"""
Anatomical tests
"""
import os
import os.path as op
import nibabel as nb

from mriqc.data import get_brainweb_1mm_normal
from mriqc.qc.anatomical import snr, snr2, snr_spectral, cjv
import numpy as np
# from numpy.testing import allclose

def test_snr():
    data = op.join(get_brainweb_1mm_normal(), 'sub-normal01')

    wmmask = op.join(get_brainweb_1mm_normal(), 'derivatives', 'volume_fraction_wht.nii.gz')
    wmdata = nb.load(wmmask).get_data().astype(np.float32)
    airmask = op.join(get_brainweb_1mm_normal(), 'derivatives', 'volume_fraction_bck.nii.gz')
    airdata = nb.load(airmask).get_data().astype(np.float32)

    ses = 'ses-pn0rf00'
    im_file = op.join(data, ses, 'anat', 'sub-normal01_%s_T1w.nii.gz' % ses)
    imdata = nb.load(im_file).get_data()

    fg_mean = np.median(imdata[wmdata > .99])

    snrs_bg = []
    snrs_wm = []
    sigmas = [0.02, 0.03, 0.05, 0.08, 0.12, 0.15, 0.2, 0.4, 0.5]

    for sigma_n in sigmas:
        test_data = imdata + np.random.normal(0.0, scale=sigma_n*fg_mean, size=imdata.shape)
        test_data[test_data < 0] = 0
        snrs_wm.append(snr(test_data, wmdata))
        snrs_bg.append(snr(test_data, wmdata, airdata))

    ref = 1./np.array(sigmas)
    err1 = ((np.array(snrs_wm) - ref) ** 2) / ref
    err2 = ((np.array(snrs_bg) - ref) ** 2) / ref
    print np.average([err1, err2], axis=1)
    return [err < 2.0 for err in np.average([err1, err2], axis=1)]


def test_snr_rayleigh():
    data = op.join(get_brainweb_1mm_normal(), 'sub-normal01')

    wmmask = op.join(get_brainweb_1mm_normal(), 'derivatives', 'volume_fraction_wht.nii.gz')
    wmdata = nb.load(wmmask).get_data().astype(np.float32)
    airmask = op.join(get_brainweb_1mm_normal(), 'derivatives', 'volume_fraction_bck.nii.gz')
    airdata = nb.load(airmask).get_data().astype(np.float32)

    ses = 'ses-pn0rf00'
    im_file = op.join(data, ses, 'anat', 'sub-normal01_%s_T1w.nii.gz' % ses)
    imdata = nb.load(im_file).get_data()

    fg_mean = np.median(imdata[wmdata > .99])

    snrs_bg = []
    snrs_wm = []
    sigmas = [0.02, 0.03, 0.05, 0.08, 0.12, 0.15, 0.2, 0.4, 0.5]

    for sigma_n in sigmas:
        test_data = imdata + np.random.rayleigh(scale=sigma_n*fg_mean, size=imdata.shape)
        test_data[test_data < 0] = 0
        snrs_wm.append(snr(test_data, wmdata))
        snrs_bg.append(snr(test_data, wmdata, airdata))

    ref = 1./np.array(sigmas)
    err1 = ((np.array(snrs_wm) - ref) ** 2) / ref
    err2 = ((np.array(snrs_bg) - ref) ** 2) / ref
    print np.average([err1, err2], axis=1)
    return [err < 2.0 for err in np.average([err1, err2], axis=1)]


def test_snr_spectral():
    data = op.join(get_brainweb_1mm_normal(), 'sub-normal01')

    wmmask = op.join(get_brainweb_1mm_normal(), 'derivatives', 'volume_fraction_wht.nii.gz')
    wmdata = nb.load(wmmask).get_data().astype(np.float32)
    airmask = op.join(get_brainweb_1mm_normal(), 'derivatives', 'volume_fraction_bck.nii.gz')
    airdata = nb.load(airmask).get_data().astype(np.float32)

    ses = 'ses-pn0rf00'
    im_file = op.join(data, ses, 'anat', 'sub-normal01_%s_T1w.nii.gz' % ses)
    imdata = nb.load(im_file).get_data()

    fg_mean = np.median(imdata[wmdata > .99])

    snrs_bg = []
    snrs_wm = []
    sigmas = [0.02, 0.03, 0.05, 0.08, 0.12, 0.15, 0.2, 0.4, 0.5]

    for sigma_n in sigmas:
        test_data = imdata + np.random.rayleigh(scale=sigma_n*fg_mean, size=imdata.shape)
        test_data[test_data < 0] = 0
        snrs_wm.append(snr_spectral(test_data, wmdata))
        snrs_bg.append(snr_spectral(test_data, wmdata, airdata))

    ref = 1./np.array(sigmas)
    err1 = ((np.array(snrs_wm) - ref) ** 2) / ref
    err2 = ((np.array(snrs_bg) - ref) ** 2) / ref

    print err1
    print err2
    return np.average([err1, err2], axis=1)


def test_snr2():
    data = op.join(get_brainweb_1mm_normal(), 'sub-normal01')

    wmmask = op.join(get_brainweb_1mm_normal(), 'derivatives', 'volume_fraction_wht.nii.gz')
    wmdata = nb.load(wmmask).get_data().astype(np.float32)
    airmask = op.join(get_brainweb_1mm_normal(), 'derivatives', 'volume_fraction_bck.nii.gz')
    airdata = nb.load(airmask).get_data().astype(np.float32)

    ses = 'ses-pn0rf00'
    im_file = op.join(data, ses, 'anat', 'sub-normal01_%s_T1w.nii.gz' % ses)
    imdata = nb.load(im_file).get_data()

    fg_mean = np.median(imdata[wmdata > .99])

    snrs_bg = []
    sigmas = [0.01, 0.03, 0.05, 0.08, 0.12, 0.15, 0.20]

    for sigma_n in sigmas:
        test_data = imdata + np.random.normal(0.0, scale=sigma_n*fg_mean, size=imdata.shape)
        test_data[test_data < 0] = 0
        snrs_bg.append(snr2(test_data, wmdata, airdata))

    ref = 1./np.array(sigmas)
    err2 = ((np.array(snrs_bg) - ref) ** 2) / ref
    print np.average(err2)
    return [err < 2.0 for err in np.average([err2], axis=1)]

def test_cjv():
    data = op.join(get_brainweb_1mm_normal(), 'sub-normal01')

    wmmask = op.join(get_brainweb_1mm_normal(), 'derivatives', 'volume_fraction_wht.nii.gz')
    wmdata = nb.load(wmmask).get_data().astype(np.float32)
    gmmask = op.join(get_brainweb_1mm_normal(), 'derivatives', 'volume_fraction_gry.nii.gz')
    gmdata = nb.load(gmmask).get_data().astype(np.float32)

    ses = 'ses-pn0rf00'
    im_file = op.join(data, ses, 'anat', 'sub-normal01_%s_T1w.nii.gz' % ses)
    imdata = nb.load(im_file).get_data()

    fg_mean = np.mean(imdata[wmdata > 0], weights=wmdata[wmdata > 0])
    cjvs = []
    sigmas = [0.01, 0.03, 0.05, 0.08, 0.12, 0.15, 0.20]
    exp_cjvs = [0.45419429982401677, 0.51149489333538289, 0.60775532593579662, 0.79450797884093927, 1.0781050744254561, 1.3013580100905036, 1.6898180485107017]

    for sigma_n in sigmas:
        test_data = imdata + np.random.normal(0.0, scale=sigma_n*fg_mean, size=imdata.shape)
        test_data[test_data < 0] = 0
        cjvs.append(cjv(test_data, wmdata, gmdata))

    return np.allclose(cjvs, exp_cjvs, rtol=.01)
