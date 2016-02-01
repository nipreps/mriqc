#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# @Author: oesteban
# @Date:   2016-01-05 11:24:05
# @Email:  code@oscaresteban.es
# @Last modified by:   oesteban
# @Last Modified time: 2016-01-18 08:36:22
""" The core module combines the existing workflows """

from nipype.pipeline import engine as pe
from nipype.interfaces import utility as niu

from .anatomical import anat_qc_workflow
from .functional import fmri_qc_workflow


def qc_workflows(name='CRN_QC', settings=None, subjects=None):
    """ The CRN quality control workflows """
    if subjects is None:
        raise RuntimeError('No subjects were provided')

    if settings is None:
        settings = {}

    wf = pe.Workflow(name=name)
    if 'work_dir' in settings.keys():
        wf.base_dir = settings['work_dir']

    outputnode = pe.Node(niu.IdentityInterface(fields=['out_group']), name='outputnode')

    skip = settings.get('skip', [])
    n_workflows = 2 - len(skip)

    merge = pe.Node(niu.Merge(n_workflows), name='merge_outputs')
    if 'anat' not in skip and subjects['anat']:
        anat_wf = anat_qc_workflow(settings=settings, sub_list=subjects['anat'])
        if settings.get('write_graph', False):
            anat_wf.write_graph()
        n_workflows += 1
        wf.connect([(anat_wf, merge, [('outputnode.out_group', 'in%d' % n_workflows)])])

    if 'func' not in skip and subjects['func']:
        func_wf = fmri_qc_workflow(settings=settings, sub_list=subjects['func'])
        if settings.get('write_graph', False):
            func_wf.write_graph()
        n_workflows += 1
        wf.connect([(func_wf, merge, [('outputnode.out_group', 'in%d' % n_workflows)])])

    wf.connect(merge, 'out', outputnode, 'out_group')

    return wf
