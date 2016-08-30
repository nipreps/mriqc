#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# @Author: oesteban
# @Date:   2016-01-05 11:29:40
# @Email:  code@oscaresteban.es
# @Last modified by:   oesteban
# @Last Modified time: 2016-08-26 14:36:31
"""
Data grabbers
"""

from mriqc.data.utils import _get_dataset_dir, _fetch_file

GOOGLEDRIVE_URL = ('https://3552243d5be815c1b09152da6525cb8fe7b900a6.googledrive.com/'
                   'host/0BxI12kyv2olZVUswazA3NkFvOXM')

def get_brainweb_1mm_normal(data_dir=None, url=None, resume=True, verbose=1):
    """Download and load the BIDS-fied brainweb 1mm normal


    :param str data_dir: path of the data directory. Used to force data storage
        in a non-standard location.
    :param str url: download URL of the dataset. Overwrite the default URL.

    """
    dataset_name = 'brainweb'
    if url is None:
        url = '{}/{}.tar'.format(GOOGLEDRIVE_URL, dataset_name)

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

    dataset_name = 'ds003_downsampled'
    if url is None:
        url = '{}/{}.tar.gz'.format(GOOGLEDRIVE_URL, dataset_name)

    data_dir = _get_dataset_dir(dataset_name, data_dir=data_dir, verbose=verbose)

    if _fetch_file(url, data_dir, filetype='tar.gz', resume=resume, verbose=verbose):
        return data_dir
    else:
        return None

def get_mni_template(data_dir=None, url=None, resume=True, verbose=1):
    """Download and load the necessary files from the mni template


    :param str data_dir: path of the data directory. Used to force data storage
        in a non-standard location.
    :param str url: download URL of the dataset. Overwrite the default URL.

    """
    dataset_name = 'mni_template'
    if url is None:
        url = '{}/{}.tar'.format(GOOGLEDRIVE_URL, dataset_name)

    data_dir = _get_dataset_dir(dataset_name, data_dir=data_dir, verbose=verbose)

    if _fetch_file(url, data_dir, filetype='tar', resume=resume, verbose=verbose,
                   md5sum='debfa882b8c301cd6d75dd769e73f727'):
        return data_dir
    else:
        return None
