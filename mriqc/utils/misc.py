#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
""" Helper functions """
from __future__ import print_function, division, absolute_import, unicode_literals

import os
from os import path as op
from glob import glob
from errno import EEXIST

import collections
import json
import pandas as pd
from io import open  # pylint: disable=W0622
from builtins import range  # pylint: disable=W0622

BIDS_COMP = collections.OrderedDict([
    ('subject_id', 'sub'), ('session_id', 'ses'), ('task_id', 'task'),
    ('acq_id', 'acq'), ('rec_id', 'rec'), ('run_id', 'run')
])

BIDS_EXPR = """\
^sub-(?P<subject_id>[a-zA-Z0-9]+)(_ses-(?P<session_id>[a-zA-Z0-9]+))?\
(_task-(?P<task_id>[a-zA-Z0-9]+))?(_acq-(?P<acq_id>[a-zA-Z0-9]+))?\
(_rec-(?P<rec_id>[a-zA-Z0-9]+))?(_run-(?P<run_id>[a-zA-Z0-9]+))?\
"""

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


def reorient_and_discard_non_steady(in_file, float32=False):
    import nibabel as nb
    import os
    import numpy as np
    import nibabel as nb
    from statsmodels.robust.scale import mad

    _, outfile = os.path.split(in_file)

    nii = nb.as_closest_canonical(nb.load(in_file))
    in_data = nii.get_data()

    # downcast to reduce space consumption and improve performance
    if float32 and np.dtype(in_data.dtype).itemsize > 4:
        in_data = in_data.astype(np.float32)

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

    nb.Nifti1Image(in_data[:, :, :, exclude_index:], nii.affine, nii.header).to_filename(outfile)
    nii.uncache()
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


def generate_pred(derivatives_dir, output_dir, mod):
    """
    Reads the metadata in the JIQM (json iqm) files and
    generates a corresponding prediction CSV table
    """

    if mod != 'T1w':
        return None

    # If some were found, generate the CSV file and group report
    out_csv = op.join(output_dir, mod + '_predicted_qa.csv')
    jsonfiles = glob(op.join(derivatives_dir, 'sub-*_%s.json' % mod))
    if not jsonfiles:
        return None

    headers = list(BIDS_COMP.keys()) + ['mriqc_pred']
    predictions = {k: [] for k in headers}

    for jsonfile in jsonfiles:
        with open(jsonfile, 'r') as jsondata:
            data = json.load(jsondata).pop('bids_meta', None)

        if data is None:
            continue

        for k in headers:
            predictions[k].append(data.pop(k, None))

    dataframe = pd.DataFrame(
        predictions).sort_values(by=list(BIDS_COMP.keys()))

    # Drop empty columns
    dataframe.dropna(axis='columns', how='all', inplace=True)

    bdits_cols = list(set(BIDS_COMP.keys()) & set(dataframe.columns.ravel()))

    # Drop duplicates
    dataframe.drop_duplicates(bdits_cols,
                              keep='last', inplace=True)

    dataframe[bdits_cols + ['mriqc_pred']].to_csv(out_csv, index=False)
    return out_csv


def generate_csv(derivatives_dir, output_dir, mod):
    """
    Generates a csv file from all json files in the derivatives directory
    """

    # If some were found, generate the CSV file and group report
    out_csv = op.join(output_dir, mod + '.csv')
    jsonfiles = glob(op.join(derivatives_dir, 'sub-*_%s.json' % mod))
    if not jsonfiles:
        return None, out_csv

    datalist = []
    comps = set(list(BIDS_COMP.keys()))
    for jsonfile in jsonfiles:
        dfentry = _read_and_save(jsonfile)

        if (dfentry is not None and dfentry['bids_meta'].get(
                'modality', 'unknown') == mod):
            metadata = dfentry.pop('bids_meta')
            id_fields = list(comps & set(list(metadata.keys())))
            for field in id_fields:
                dfentry[field] = metadata[field]
            dfentry.pop('provenance', None)
            datalist.append(dfentry)

    dataframe = pd.DataFrame(datalist)
    cols = dataframe.columns.tolist()  # pylint: disable=no-member

    # Generate bids_fields with order (#516)
    bids_fields = [field for field in list(BIDS_COMP.keys()) if field in cols]

    # Sort the dataframe, with failsafe if pandas version is too old
    try:
        dataframe = dataframe.sort_values(by=bids_fields)
    except AttributeError:
        #pylint: disable=E1101
        dataframe = dataframe.sort(columns=bids_fields)

    # Drop duplicates
    try:
        #pylint: disable=E1101
        dataframe.drop_duplicates(bids_fields, keep='last', inplace=True)
    except TypeError:
        #pylint: disable=E1101
        dataframe.drop_duplicates(['subject_id', 'session_id', 'run_id'], take_last=True,
                                  inplace=True)

    ordercols = bids_fields + sorted(list(set(cols) - set(bids_fields)))
    dataframe[ordercols].to_csv(out_csv, index=False)
    return dataframe, out_csv


def _read_and_save(in_file):
    with open(in_file, 'r') as jsondata:
        data = json.load(jsondata)
    return data if data else None


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
