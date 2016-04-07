#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# @Author: oesteban
# @Date:   2016-01-05 11:29:40
# @Email:  code@oscaresteban.es
# @Last modified by:   oesteban
# @Last Modified time: 2016-04-06 17:29:07
"""
Anatomical tests
"""
import os
import os.path as op
import nibabel as nb

from mriqc.data import get_brainweb_1mm_normal
from mriqc.qc.anatomical import snr
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

    fg_mean = np.average(imdata[wmdata > 0], weights=wmdata[wmdata > 0])

    snrs = []
    sigmas = [0.01, 0.03, 0.05, 0.08, 0.12, 0.15, 0.20]
    expected_snr = [39.9, 33.7, 26.4, 18.9, 13.3, 10.9, 8.3]

    for sigma_n in sigmas:
        test_data = imdata + np.random.normal(0.0, scale=sigma_n*fg_mean, size=imdata.shape)
        test_data[test_data < 0] = 0
        snrs.append(snr(test_data, wmdata, airdata))

    return np.allclose(snrs, expected_snr, rtol=.01)


#
#    sessions = [op.basename(ses) for ses in os.listdir(data)
#                if op.isdir(op.join(data, ses))]
#
#    test_results = []
#    for ses in sorted(sessions):
#        im_file = op.join(data, ses, 'anat', 'sub-normal01_%s_T1w.nii.gz' % ses)
#        imdata = nb.load(im_file).get_data()
#        print ses, 1. / snr(imdata, wmdata)
#        # test_results.append(assert_almost_equal(snr(imdata, wmdata), 0.04))
#
#    return test_results

