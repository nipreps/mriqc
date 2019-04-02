#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# @Author: oesteban
# @Date:   2016-01-05 11:24:05
# @Email:  code@oscaresteban.es
""" The core module combines the existing workflows """

import os
from .anatomical import anat_qc_workflow
from .functional import fmri_qc_workflow


def build_workflow(dataset, mod, settings=None):
    """ Multi-subject anatomical workflow wrapper """

    settings["biggest_file_size_gb"] = _get_biggest_file_size_gb(dataset)

    if mod not in ('T1w', 'T2w', 'bold'):
        raise NotImplementedError('Unknown workflow type "%s"' % mod)

    if mod == 'bold':
        workflow = fmri_qc_workflow(dataset, settings=settings)
    else:
        workflow = anat_qc_workflow(dataset, mod=mod, settings=settings)

    workflow.base_dir = str(settings['work_dir'])
    if settings.get('write_graph', False):
        workflow.write_graph()
    return workflow


def _get_biggest_file_size_gb(files):
    max_size = 0
    for file in files:
        size = os.path.getsize(file) / (1024 ** 3)
        if size > max_size:
            max_size = size
    return max_size
