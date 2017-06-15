#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# @Author: oesteban
# @Date:   2016-01-05 11:24:05
# @Email:  code@oscaresteban.es
""" The core module combines the existing workflows """
from __future__ import print_function, division, absolute_import, unicode_literals

import os
from .anatomical import anat_qc_workflow
from .functional import fmri_qc_workflow


def build_workflow(dataset, mod, settings=None):
    """ Multi-subject anatomical workflow wrapper """

    settings["biggest_file_size_gb"] = _get_biggest_file_size_gb(dataset)

    if mod == 'bold':
        workflow = fmri_qc_workflow(dataset, settings=settings)
    elif mod == 'T1w':
        workflow = anat_qc_workflow(dataset, mod='T1w', settings=settings)
    elif mod == 'T2w':
        workflow = anat_qc_workflow(dataset, mod='T2w', settings=settings)
    else:
        raise NotImplementedError('Unknown workflow type "%s"' % mod)

    workflow.base_dir = settings['work_dir']
    if settings.get('write_graph', False):
        workflow.write_graph()
    return workflow

def _get_biggest_file_size_gb(files):
    max_size = 0
    for file in files:
        size = os.path.getsize(file)/(1024*1024*1024)
        if size > max_size:
            max_size = size
    return max_size
