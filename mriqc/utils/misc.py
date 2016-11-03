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
from builtins import next, range  # pylint: disable=W0622

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


def reorient(in_file):
    import nibabel as nb
    import os
    _, outfile = os.path.split(in_file)
    nii = nb.as_closest_canonical(nb.load(in_file))
    nii.to_filename(outfile)
    return os.path.abspath(outfile)


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

def bids_getfile(bids_dir, data_type, subject_id, session_id=None, run_id=None):
    """
    A simple function to select files from a BIDS structure

    Example::

    >>> from niworkflows.data import get_ds003_downsampled
    >>> bids_getfile(get_ds003_downsampled(), 'anat', '05') #doctest: +ELLIPSIS +IGNORE_UNICODE
    '...ds003_downsampled/sub-05/anat/sub-05_T1w.nii.gz'

    """
    import os.path as op
    import glob

    if data_type == 'anat':
        scan_type = 'T1w'

    if data_type == 'func':
        scan_type = 'bold'

    out_file = op.join(bids_dir, subject_id)

    onesession = (session_id is None or session_id == '0')
    onerun = (run_id is None or run_id == '0')

    if onesession:
        if onerun:
            pattern = op.join(out_file, data_type, '{}_*{}.nii*'.format(subject_id, scan_type))
        else:
            pattern = op.join(out_file, data_type, '{}_*{}_{}.nii*'.format(subject_id, run_id, scan_type))

    else:
        if onerun:
            pattern = op.join(out_file, session_id, data_type,
                              '{}_{}_*{}.nii*'.format(subject_id, session_id, scan_type))
        else:
            pattern = op.join(out_file, session_id, data_type,
                              '{}_{}*_{}_{}.nii*'.format(subject_id, session_id, run_id, scan_type))

    results = glob.glob(pattern)

    if not results:
        raise RuntimeError(
            'No file found with this pattern: "{}", finding '
            'BIDS dataset coordinates are ({}, {}, {})'.format(pattern, subject_id, session_id, run_id))

    return results[0]


def bids_scan_file_walker(dataset=".", include_types=None, warn_no_files=False):
    """
    Traverse a BIDS dataset and provide a generator interface
    to the imaging files contained within.

    :author: @chrisfilo

    https://github.com/preprocessed-connectomes-project/quality-assessment-prot\
ocol/blob/master/scripts/qap_bids_data_sublist_generator.py

    :param str dataset: path to the BIDS dataset folder.

    :param list(str) include_types: a list of the scan types (i.e.
      subfolder names) to include in the results. Can be any combination
      of "func", "anat", "fmap", "dwi".

    :param bool warn_no_files: issue a warning if no imaging files are found
      for a subject or a session.

    :return: a list containing, for each .nii or .nii.gz file found, the BIDS
      identifying tokens and their values. If a file doesn't have an
      identifying token its key will be None.

    """
    import os
    import os.path as op
    from glob import glob

    from warnings import warn

    def _no_files_warning(folder):
        if not warn_no_files:
            return
        warn("No files of requested type(s) found in scan folder: {}"
            .format(folder), RuntimeWarning, stacklevel=1)

    def _walk_dir_for_prefix(target_dir, prefix):
        return [x for x in next(os.walk(target_dir))[1]
                if x.startswith(prefix)]

    def _tokenize_bids_scan_name(scanfile):
        scan_basename = op.splitext(op.split(scanfile)[1])[0]
        # .nii.gz will have .nii leftover
        scan_basename = scan_basename.replace(".nii", "")
        file_bits = scan_basename.split('_')

        # BIDS with non ses-* subfolders given default
        # "0" ses.
        file_tokens = {'scanfile': scanfile,
                       'sub': None, 'ses': '0',
                       'acq': None, 'rec': None,
                       'run': None, 'task': None,
                       'modality': file_bits[-1]}
        for bit in file_bits:
            for key in list(file_tokens.keys()):
                if bit.startswith(key):
                    file_tokens[key] = bit

        return file_tokens

    #########

    if include_types is None:
        # include all scan types by default
        include_types = ['func', 'anat', 'fmap', 'dwi']

    subjects = _walk_dir_for_prefix(dataset, 'sub-')
    if len(subjects) == 0:
        raise GeneratorExit("No BIDS subjects found to examine.")

    # for each subject folder, look for scans considering explicitly
    # defined sessions or the implicit "0" case.
    for subject in subjects:
        subj_dir = op.join(dataset, subject)

        sessions = _walk_dir_for_prefix(subj_dir, 'ses-')

        for scan_type in include_types:
            # seems easier to consider the case of multi-session vs.
            # single session separately?
            if len(sessions) > 0:
                subject_sessions = [op.join(subject, x)
                                    for x in sessions]
            else:
                subject_sessions = [subject]

            for session in subject_sessions:

                scan_files = glob(op.join(
                    dataset, session, scan_type,
                    '*.nii*'))

                if len(scan_files) == 0:
                    _no_files_warning(session)

                for scan_file in scan_files:
                    yield _tokenize_bids_scan_name(scan_file)


def gather_bids_data(dataset_folder, subject_inclusion=None, include_types=None):
    """ Extract data from BIDS root folder """
    import os
    import os.path as op
    from six import string_types
    import yaml
    from glob import glob

    sub_dict = {}
    inclusion_list = []

    if include_types is None:
        include_types = ['anat', 'func']

    # create subject inclusion list
    if subject_inclusion is not None and isinstance(subject_inclusion, string_types):
        with open(subject_inclusion, "r") as f:
            inclusion_list = f.readlines()
        # remove any /n's
        inclusion_list = [s.strip() for s in inclusion_list]

    if subject_inclusion is not None and isinstance(subject_inclusion, list):
        inclusion_list = []
        for s in subject_inclusion:
            if not s.startswith('sub-'):
                s = 'sub-' + s
            inclusion_list.append(s)

    sub_dict = {'anat': [], 'func': []}

    bids_inventory = bids_scan_file_walker(dataset_folder,
                                           include_types=include_types)
    for bidsfile in sorted(bids_inventory,
                           key=lambda f: f['scanfile']):

        if subject_inclusion is not None:
            if bidsfile['sub'] not in inclusion_list:
                continue

        # implies that other anatomical modalities might be
        # analyzed down the road.
        if bidsfile['modality'] in ['T1w']:  # ie, anatomical
            scan_key = '0'
            if bidsfile['run'] is not None:
                # TODO: consider multiple acq/recs
                scan_key = bidsfile['run']
            sub_dict['anat'].append(
                (bidsfile['sub'], bidsfile['ses'], scan_key))

        elif bidsfile['modality'] in ['bold']:  # ie, functional
            scan_key = bidsfile['task']
            if bidsfile['acq'] is not None:
                scan_key += '_' + bidsfile['acq']
            if bidsfile['run'] is not None:
                # TODO: consider multiple acq/recs
                scan_key += '_' + bidsfile['run']
            if scan_key is None:
                scan_key = 'func0'
            sub_dict['func'].append(
                (bidsfile['sub'], bidsfile['ses'], scan_key))

    if len(include_types) == 1:
        return sub_dict[include_types[0]]

    return sub_dict


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
        raise RuntimeError('No QC-json files found to generate QC table')

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


