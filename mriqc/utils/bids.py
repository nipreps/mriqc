#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""PyBIDS tooling"""
from collections import defaultdict

DEFAULT_TYPES = ['bold', 'T1w', 'T2w']


def collect_bids_data(layout, participant_label=None, session=None, run=None,
                      task=None, bids_type=None):
    """Get files in dataset"""

    bids_type = bids_type or DEFAULT_TYPES
    if not isinstance(bids_type, (list, tuple)):
        bids_type = [bids_type]

    basequery = {
        'subject': participant_label,
        'session': session,
        'task': task,
        'run': run,
    }
    # Filter empty lists, strings, zero runs, and Nones
    basequery = {k: v for k, v in basequery.items() if v}

    # Start querying
    imaging_data = defaultdict(list, {})
    for btype in bids_type:
        imaging_data[btype] = layout.get(
            suffix=btype,
            return_type='file',
            extensions=['nii', 'nii.gz'],
            **basequery)

    return imaging_data
