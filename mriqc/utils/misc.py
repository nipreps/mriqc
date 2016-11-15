#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
""" Helper functions """
from __future__ import print_function, division, absolute_import, unicode_literals

import os
from os import path as op
from errno import EEXIST

import collections
import json
import pandas as pd
from io import open  # pylint: disable=W0622
from builtins import range  # pylint: disable=W0622

def split_ext(in_file, out_file=None):
    import os.path as op
    if out_file is None:
        fname, ext = op.splitext(op.basename(in_file))
        if ext == '.gz':
            fname, ext2 = op.splitext(fname)
            ext = ext2 + ext
        return fname, ext
    else:
        return split_ext(out_file)


def reorient_and_discard_non_steady(in_file):
    import nibabel as nb
    import os
    import numpy as np
    import nibabel as nb
    from statsmodels.robust.scale import mad

    _, outfile = os.path.split(in_file)

    nii = nb.as_closest_canonical(nb.load(in_file))
    in_data = nii.get_data()
    data = in_data[:, :, :, :50]
    timeseries = data.max(axis=0).max(axis=0).max(axis=0)
    outlier_timecourse = (timeseries - np.median(timeseries)) / mad(
        timeseries)
    exclude_index = 0
    for i in range(10):
        if outlier_timecourse[i] > 10:
            exclude_index += 1
        else:
            break

    nb.Nifti1Image(in_data[:, :, :, exclude_index:], nii.affine).to_filename(outfile)
    return exclude_index, os.path.abspath(outfile)

def check_folder(folder):
    if not op.exists(folder):
        try:
            os.makedirs(folder)
        except OSError as exc:
            if not exc.errno == EEXIST:
                raise
    return folder

def reorder_csv(csv_file, out_file=None):
    """
    Put subject, session and scan in front of csv file

    :param str csv_file: the input csv file
    :param str out_file: if provided, a new csv file is created

    :return: the path to the file with the columns reordered


    """
    if isinstance(csv_file, list):
        csv_file = csv_file[-1]

    if out_file is None:
        out_file = csv_file

    dataframe = pd.read_csv(csv_file)
    cols = dataframe.columns.tolist()  # pylint: disable=no-member
    try:
        cols.remove('Unnamed: 0')
    except ValueError:
        # The column does not exist
        pass

    for col in ['scan', 'session', 'subject']:
        cols.remove(col)
        cols.insert(0, col)

    dataframe[cols].to_csv(out_file)
    return out_file


def rotate_files(fname):
    """A function to rotate file names"""
    import glob
    import os
    import os.path as op

    name, ext = op.splitext(fname)
    if ext == '.gz':
        name, ext2 = op.splitext(fname)
        ext = ext2 + ext

    if not op.isfile(fname):
        return

    prev = glob.glob('{}.*{}'.format(name, ext))
    prev.insert(0, fname)
    prev.append('{0}.{1:d}{2}'.format(name, len(prev) - 1, ext))
    for i in reversed(list(range(1, len(prev)))):
        os.rename(prev[i-1], prev[i])


def bids_path(subid, sesid=None, runid=None, prefix=None, out_path=None, ext='json'):
    import os.path as op
    fname = '{}'.format(subid)
    if prefix is not None:
        if not prefix.endswith('_'):
            prefix += '_'
        fname = prefix + fname
    if sesid is not None:
        fname += '_ses-{}'.format(sesid)
    if runid is not None:
        fname += '_run-{}'.format(runid)

    if out_path is not None:
        fname = op.join(out_path, fname)
    return op.abspath(fname + '.' + ext)


def generate_csv(jsonfiles, out_fname):
    """
    Generates a csv file from all json files in the derivatives directory
    """
    datalist = []
    errorlist = []

    if not jsonfiles:
        return None

    for jsonfile in jsonfiles:
        dfentry = _read_and_save(jsonfile)
        if dfentry is not None:
            if 'exec_error' not in list(dfentry.keys()):
                datalist.append(dfentry)
            else:
                errorlist.append(dfentry['subject_id'])

    dataframe = pd.DataFrame(datalist)
    cols = dataframe.columns.tolist()  # pylint: disable=no-member

    reorder = []
    for field in ['run', 'session', 'subject']:
        for col in cols:
            if col.startswith(field):
                reorder.append(col)

    for col in reorder:
        cols.remove(col)
        cols.insert(0, col)

    if 'mosaic_file' in cols:
        cols.remove('mosaic_file')

    # Sort the dataframe, with failsafe if pandas version is too old
    try:
        dataframe = dataframe.sort_values(by=['subject_id', 'session_id', 'run_id'])
    except AttributeError:
        #pylint: disable=E1101
        dataframe = dataframe.sort(columns=['subject_id', 'session_id', 'run_id'])

    # Drop duplicates
    try:
        #pylint: disable=E1101
        dataframe.drop_duplicates(['subject_id', 'session_id', 'run_id'], keep='last',
                                  inplace=True)
    except TypeError:
        #pylint: disable=E1101
        dataframe.drop_duplicates(['subject_id', 'session_id', 'run_id'], take_last=True,
                                  inplace=True)
    dataframe[cols].to_csv(out_fname, index=False)
    return dataframe, errorlist


def _read_and_save(in_file):
    with open(in_file, 'r') as jsondata:
        values = _flatten(json.load(jsondata))
        return values
    return None


def _flatten(in_dict, parent_key='', sep='_'):
    items = []
    for k, val in list(in_dict.items()):
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(val, collections.MutableMapping):
            items.extend(list(_flatten(val, new_key, sep=sep).items()))
        else:
            items.append((new_key, val))
    return dict(items)


def _flatten_dict(indict):
    out_qc = {}
    for k, value in list(indict.items()):
        if not isinstance(value, dict):
            out_qc[k] = value
        else:
            for subk, subval in list(value.items()):
                if not isinstance(subval, dict):
                    out_qc['_'.join([k, subk])] = subval
                else:
                    for ssubk, ssubval in list(subval.items()):
                        out_qc['_'.join([k, subk, ssubk])] = ssubval
    return out_qc
