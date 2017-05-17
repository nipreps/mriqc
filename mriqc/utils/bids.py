#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""PyBIDS tooling"""
from __future__ import print_function, division, absolute_import, unicode_literals

from copy import deepcopy
from bids.grabbids import BIDSLayout

from builtins import str, bytes

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

    # Set modalities
    if modalities is None:
        modalities = deepcopy(DEFAULT_MODALITIES)

    if session:
        for mod in modalities:
            queries[mod]['session'] = [session]

    if run:
        for mod in modalities:
            queries[mod]['run'] = run

    if task:
        if isinstance(task, list) and len(task) == 1:
            task = task[0]
        queries['bold']['task'] = task

    # Set participants
    if participant_label is not None:
        if isinstance(participant_label, (bytes, str)):
            participant_label = [participant_label]

        participant_label = ['{}'.format(sub) for sub in participant_label]
        participant_label = [sub[4:] if sub.startswith('sub-') else sub
                             for sub in participant_label]
        participant_label = [sub[:-1] if sub.endswith('*') else (sub + '$')
                             for sub in participant_label]
        participant_label = [sub[1:] if sub.startswith('*') else ('^' + sub)
                             for sub in participant_label]

        # For some reason, outer subject ids are filtered out
        participant_label.insert(0, 'null')
        participant_label.append('null')
        for key in queries.keys():
            queries[key]['subject'] = 'sub-\\(' + '|'.join(participant_label) + '\\){1}'

    # Start querying
    imaging_data = {}
    for mod in modalities:
        imaging_data[mod] = [x.filename for x in layout.get(**queries[mod])]

    return imaging_data
