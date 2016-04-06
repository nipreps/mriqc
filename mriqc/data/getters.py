#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# @Author: oesteban
# @Date:   2016-01-05 11:29:40
# @Email:  code@oscaresteban.es
# @Last modified by:   oesteban
# @Last Modified time: 2016-04-06 12:05:34
"""
Data grabbers
"""

from .utils import _get_dataset_dir, _fetch_file


def get_brainweb_1mm_normal(data_dir=None, url=None, resume=True, verbose=1):
    """Download and load the BIDS-fied brainweb 1mm normal


    :param str data_dir: path of the data directory. Used to force data storage
        in a non-standard location.
    :param str url: download URL of the dataset. Overwrite the default URL.

    """
    if url is None:
        url = "https://googledrive.com/host/0BxI12kyv2olZbVlRZUVmd3FaSzg"

    dataset_name = 'brainweb'
    data_dir = _get_dataset_dir(dataset_name, data_dir=data_dir, verbose=verbose)

    if _fetch_file(url, data_dir, filetype='tar', resume=resume, verbose=verbose,
                   md5sum='e338e6e1db3584eb78b00f2c6402ff28'):
        return data_dir
    else:
        return None

def get_ds003_downsampled(data_dir=None, url=None, resume=True, verbose=1):
    """Download and load the BIDS-fied ds003_downsampled


    :param str data_dir: path of the data directory. Used to force data storage
        in a non-standard location.
    :param str url: download URL of the dataset. Overwrite the default URL.

    """
    if url is None:
        url = "https://googledrive.com/host/0B2JWN60ZLkgkMEw4bW5VUUpSdFU/ds003_downsampled.tar"

    dataset_name = 'ds003_downsampled'
    data_dir = _get_dataset_dir(dataset_name, data_dir=data_dir, verbose=verbose)

    if _fetch_file(url, data_dir, filetype='tar', resume=resume, verbose=verbose):
        return data_dir
    else:
        return None

