#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""PyBIDS tooling"""
from __future__ import print_function, division, absolute_import, unicode_literals

from copy import deepcopy
from bids.grabbids import BIDSLayout

DEFAULT_MODALITIES = ['bold', 'T1w', 'T2w']
DEFAULT_QUERIES = {
    'bold': {'modality': 'func', 'type': 'bold', 'extensions': ['nii', 'nii.gz']},
    'T1w': {'modality': 'anat', 'type': 'T1w', 'extensions': ['nii', 'nii.gz']},
    'T2w': {'modality': 'anat', 'type': 'T2w', 'extensions': ['nii', 'nii.gz']}
}


def collect_bids_data(dataset, participant_label=None, session=None, run=None,
                      queries=None, task=None, modalities=None):
    """Get files in dataset"""

    # Start a layout
    layout = BIDSLayout(dataset)

    # Set queries
    if queries is None:
        queries = deepcopy(DEFAULT_QUERIES)

    if participant_label:
        subjects = ["{}".format(sub) for sub in participant_label]
        subjects = [sub[4:] if sub.startswith('sub-') else sub
                    for sub in subjects]
        subjects = ["{}[a-zA-Z0-9]*".format(sub[:-1]) if sub.endswith('*') else (sub + '$')
                    for sub in subjects]
        subjects = ["[a-zA-Z0-9]*{}".format(sub[1:]) if sub.startswith('*') else ('^' + sub)
                    for sub in subjects]

        # For some reason, outer subject ids are filtered out
        subjects.insert(0, 'null')
        subjects.append('null')

        for key in queries.keys():
            queries[key]['subject'] = 'sub-\\(' + '|'.join(subjects) + '\\){1}'

    if session:
        sessions = ["{}".format(ses) for ses in session]
        sessions = [ses[4:] if ses.startswith('ses-') else ses
                    for ses in sessions]
        sessions = ["{}[a-zA-Z0-9]*".format(ses[:-1]) if ses.endswith('*') else (ses + '$')
                    for ses in sessions]
        sessions = ["[a-zA-Z0-9]*{}".format(ses[1:]) if ses.startswith('*') else ('^' + ses)
                    for ses in sessions]

        # For some reason, outer session ids are filtered out
        sessions.insert(0, 'null')
        sessions.append('null')

        for key in queries.keys():
            queries[key]['session'] = 'ses-\\(' + '|'.join(sessions) + '\\){1}'

    if run:
        runs = ["{}".format(run) for run in run]
        runs = [run[4:] if run.startswith('run-') else run
                for run in runs]
        runs = ["{}\\d*".format(run[:-1]) if run.endswith('*') else run
                for run in runs]
        runs = ["\\d*{}".format(run[1:]) if run.startswith('*') else run
                for run in runs]

        # For some reason, outer session ids are filtered out
        runs.insert(0, 'null')
        runs.append('null')

        # For some reason, outer subject ids are filtered out
        participant_label.insert(0, 'null')
        participant_label.append('null')

        for key in queries.keys():
            queries[key]['run'] = '\\(run-' + '|'.join(runs) + '\\){1}'

    if task:
        if isinstance(task, list) and len(task) == 1:
            task = task[0]
        queries['bold']['task'] = task

    # Set modalities
    if not modalities:
        modalities = deepcopy(DEFAULT_MODALITIES)

    # Start querying
    imaging_data = {}
    for mod in modalities:
        imaging_data[mod] = [x.filename for x in layout.get(**queries[mod])]

    return imaging_data
