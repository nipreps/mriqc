#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# @Author: oesteban
# @Date:   2016-01-05 11:24:05
# @Email:  code@oscaresteban.es
# @Last modified by:   oesteban
# @Last Modified time: 2016-05-03 11:23:23
""" The core module combines the existing workflows """
from six import string_types
from .anatomical import anat_qc_workflow
from .functional import fmri_qc_workflow

def ms_anat(settings=None, subject_id=None, session_id=None, run_id=None):
    if subject_id is None:
        raise NotImplementedError

    if isinstance(subject_id, string_types):
        workflow = anat_qc_workflow(name='mriqc_sub_' + subject_id, settings=settings)
        workflow.inputs.inputnode.bids_root = settings['bids_root']
        workflow.inputs.inputnode.subject_id = subject_id
        workflow.inputs.inputnode.session_id = 'single_session' if session_id is None else session_id
        workflow.inputs.inputnode.run_id = 'single_run' if run_id is None else run_id
    return workflow



def qc_workflows(settings=None, subjects=None):
    """ The CRN quality control workflows """
    if subjects is None:
        raise RuntimeError('No subjects were provided')

    if settings is None:
        settings = {}

    skip = settings.get('skip', [])
    anat_wf = None
    if 'anat' not in skip and subjects['anat']:
        anat_wf = anat_qc_workflow(settings=settings)
        if 'work_dir' in settings.keys():
            anat_wf.base_dir = settings['work_dir']

        if settings.get('write_graph', False):
            anat_wf.write_graph()

    func_wf = None
    if 'func' not in skip and subjects['func']:
        func_wf = fmri_qc_workflow(settings=settings, sub_list=subjects['func'])
        if 'work_dir' in settings.keys():
            func_wf.base_dir = settings['work_dir']

        if settings.get('write_graph', False):
            func_wf.write_graph()

    return anat_wf, func_wf
