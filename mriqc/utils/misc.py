#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
""" Helper functions """


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
        warn("No files of requested type(s) found in scan folder: %s"
             % folder, RuntimeWarning, stacklevel=1)

    def _walk_dir_for_prefix(target_dir, prefix):
        return [x for x in next(os.walk(target_dir))[1]
                if x.startswith(prefix)]

    def _tokenize_bids_scan_name(scanfile):
        scan_basename = op.splitext(op.split(scanfile)[1])[0]
        # .nii.gz will have .nii leftover
        scan_basename = scan_basename.replace(".nii", "")
        file_bits = scan_basename.split('_')

        # BIDS with non ses-* subfolders given default
        # "single_session" ses.
        file_tokens = {'scanfile': scanfile,
                       'sub': None, 'ses': "single_session",
                       'acq': None, 'rec': None,
                       'run': None, 'task': None,
                       'modality': file_bits[-1]}
        for bit in file_bits:
            for key in file_tokens.keys():
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
    # defined sessions or the implicit "single_session" case.
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


def gather_bids_data(dataset_folder, subject_inclusion=None, scan_type=None):
    """ Extract data from BIDS root folder """
    import os
    import os.path as op
    import yaml
    from glob import glob

    sub_dict = {}
    inclusion_list = []

    if scan_type is None:
        scan_type = 'functional anatomical'

    # create subject inclusion list
    if subject_inclusion is not None:
        with open(subject_inclusion, "r") as f:
            inclusion_list = f.readlines()
        # remove any /n's
        inclusion_list = [s.strip() for s in inclusion_list]

    sub_dict = {'anat': [], 'func': []}

    bids_inventory = bids_scan_file_walker(dataset_folder,
                                           include_types=['anat', 'func'])
    for bidsfile in sorted(bids_inventory,
                           key=lambda f: f['scanfile']):

        if subject_inclusion is not None:
            if bidsfile['sub'] not in inclusion_list:
                continue

        # implies that other anatomical modalities might be
        # analyzed down the road.
        if bidsfile['modality'] in ['T1w']:  # ie, anatomical
            scan_key = bidsfile['modality']
            if bidsfile['run'] is not None:
                # TODO: consider multiple acq/recs
                scan_key += '_' + bidsfile['run']
            sub_dict['anat'].append(
                (bidsfile['sub'], bidsfile['ses'], scan_key, op.abspath(bidsfile['scanfile'])))

        elif bidsfile['modality'] in ['bold']:  # ie, functional
            scan_key = bidsfile['task']
            if bidsfile['run'] is not None:
                # TODO: consider multiple acq/recs
                scan_key += '_' + bidsfile['run']
            if scan_key is None:
                scan_key = 'func_1'
            sub_dict['func'].append(
                (bidsfile['sub'], bidsfile['ses'], scan_key, op.abspath(bidsfile['scanfile'])))

    return sub_dict


def reorder_csv(csv_file, out_file=None):
    """
    Put subject, session and scan in front of csv file

    :param str csv_file: the input csv file
    :param str out_file: if provided, a new csv file is created

    :return: the path to the file with the columns reordered


    """
    import pandas as pd
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

    prev = glob.glob('%s.*%s' % (name, ext))
    prev.insert(0, fname)
    prev.append('%s.%d%s' % (name, len(prev) - 1, ext))
    for i in reversed(range(1, len(prev))):
        os.rename(prev[i-1], prev[i])
