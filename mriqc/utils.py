#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

# Original author: @chrisfilo
# https://github.com/preprocessed-connectomes-project/quality-assessment-prot
# ocol/blob/master/scripts/qap_bids_data_sublist_generator.py


def gather_bids_data(dataset_folder, subject_inclusion=None, scan_type=None):
    import os
    import os.path as op
    import yaml, re
    from glob import glob

    sub_dict = {}
    inclusion_list = []

    subject_ids = [x for x in next(os.walk(dataset_folder))[1]
                   if x.startswith("sub-")]

    if scan_type is None:
        scan_type = 'functional anatomical'

    get_anat = 'anatomical' in scan_type
    get_func = 'functional' in scan_type

    if not subject_ids:
        raise Exception("This does not appear to be a BIDS dataset.")

    # create subject inclusion list
    if subject_inclusion is not None:
        with open(subject_inclusion, "r") as f:
            inclusion_list = f.readlines()
        # remove any /n's
        inclusion_list = map(lambda s: s.strip(), inclusion_list)
        subject_ids = set(subject_ids).intersection(inclusion_list)

    sub_dict = {'anat': [], 'func': []}

    for subject_id in sorted(subject_ids):
        # TODO: implement multisession support
        ssid = 'session_1'

        anatomical_scans = []
        if get_anat:
            anatomical_scans = sorted(glob(op.join(
                dataset_folder, subject_id, "anat",
                "%s_*T1w.nii.gz" % subject_id, )))

        for i, scan in enumerate(anatomical_scans):
            scid = 'anat_%d' % (i+1)
            spath = op.abspath(scan)
            sub_dict['anat'].append((subject_id, ssid, scid, spath))

        functional_scans = []
        if get_func:
            functional_scans = sorted(glob(op.join(
                dataset_folder, subject_id, "func",
                "%s_*bold*.nii.gz" % subject_id, )))

        for i, scan in enumerate(functional_scans):
            taskinfo = re.search(r'.+?task-(([^_]+)(_run-([0-9]+))?)_bold.nii(?:.gz)?$', scan)
            if taskinfo is None:
                # warn user of potential nonstandard bids layout?
                scid = 'func_%d' % (i+1)
            else:
                scid = taskinfo.group(1)
            spath = op.abspath(scan)
            sub_dict['func'].append((subject_id, ssid, scid, spath))

    return sub_dict


def reorder_csv(csv_file, out_file=None):
    import pandas as pd
    if isinstance(csv_file, list):
        csv_file = csv_file[-1]

    if out_file is None:
        out_file = csv_file

    df = pd.read_csv(csv_file)
    cols = df.columns.tolist()
    try:
        cols.remove('Unnamed: 0')
    except ValueError:
        # The column does not exist
        pass

    for v in ['scan', 'session', 'subject']:
        cols.remove(v)
        cols.insert(1, v)
    df[cols].to_csv(out_file)
    return out_file
