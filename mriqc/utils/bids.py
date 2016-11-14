#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""PyBIDS tooling"""
from __future__ import print_function, division, absolute_import, unicode_literals

from copy import deepcopy
from bids.grabbids import BIDSLayout

DEFAULT_MODALITIES = ['func', 't1w']
DEFAULT_QUERIES = {
    'func': {'modality': 'func', 'type': 'bold', 'ext': 'nii'},
    't1w': {'type': 'T1w', 'ext': 'nii'}
}

def collect_bids_data(dataset, participant_label=None, session=None, run=None,
                      queries=None, modalities=None):
    """Get files in dataset"""

    # Start a layout
    layout = BIDSLayout(dataset)

    # Find all sessions
    if session:
        session_list = [session]
    else:
        session_list = layout.unique('session')
        if session_list == []:
            session_list = [None]

    # Find all runs
    if run:
        run_list = [run]
    else:
        run_list = layout.unique('run')
        if run_list == []:
            run_list = [None]

    # Set modalities
    if modalities is None:
        modalities = deepcopy(DEFAULT_MODALITIES)

    # Set queries
    if queries is None:
        queries = deepcopy(DEFAULT_QUERIES)

    # Set participants
    if participant_label is not None:
        if not isinstance(participant_label, list):
            for key in queries.keys():
                queries[key]['subject'] = participant_label
        else:
            participant_label = ['{}'.format(sub) for sub in participant_label]
            participant_label = [sub[4:] if sub.startswith('sub-') else sub
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
