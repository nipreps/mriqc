#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# @Author: oesteban
# @Date:   2016-01-05 11:29:40
# @Email:  code@oscaresteban.es
# @Last modified by:   oesteban
# @Last Modified time: 2016-04-06 16:33:35
"""
Data grabbers
"""

from mriqc.data.utils import _get_dataset_dir, _fetch_file


def get_brainweb_1mm_normal(data_dir=None, url=None, resume=True, verbose=1):
    """Download and load the BIDS-fied brainweb 1mm normal


    :param str data_dir: path of the data directory. Used to force data storage
        in a non-standard location.
    :param str url: download URL of the dataset. Overwrite the default URL.

    """
    if url is None:
        url = "https://googledrive.com/host/0BxI12kyv2olZZkhrUzZLbExKRzQ"

    dataset_name = 'brainweb'
    data_dir = _get_dataset_dir(dataset_name, data_dir=data_dir, verbose=verbose)

    if _fetch_file(url, data_dir, filetype='tar', resume=resume, verbose=verbose,
                   md5sum='384263fbeadc8e2cca92ced98f224c4b'):
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

def get_mni_template(data_dir=None, url=None, resume=True, verbose=1):
    """Download and load the necessary files from the mni template


    :param str data_dir: path of the data directory. Used to force data storage
        in a non-standard location.
    :param str url: download URL of the dataset. Overwrite the default URL.

    """
    if url is None:
        url = "http://googledrive.com/host/0BxI12kyv2olZdzRDUnBPYWZGZk0"

    dataset_name = 'mni_template'
    data_dir = _get_dataset_dir(dataset_name, data_dir=data_dir, verbose=verbose)

    if _fetch_file(url, data_dir, filetype='tar', resume=resume, verbose=verbose,
                   md5sum='debfa882b8c301cd6d75dd769e73f727'):
        return data_dir
    else:
        return None
