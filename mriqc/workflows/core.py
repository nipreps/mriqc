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
from ..interfaces.viz import Report
from ..utils import reorder_csv


def qc_workflows(name='CRN_QC', settings=None, subjects=None):
    """ The CRN quality control workflows """

    def _back(inlist):
        if isinstance(inlist, list):
            return inlist[-1]
        else:
            return inlist

    if subjects is None:
        raise RuntimeError('No subjects were provided')

    if settings is None:
        settings = {}

    wf = pe.Workflow(name=name)
    if 'work_dir' in settings.keys():
        wf.base_dir = settings['work_dir']

    outputnode = pe.Node(niu.IdentityInterface(fields=['anat_csv', 'func_csv']), name='outputnode')

    skip = settings.get('skip', [])
    if 'anat' not in skip and subjects['anat']:
        anat_wf = anat_qc_workflow(settings=settings)
        if settings.get('write_graph', False):
            anat_wf.write_graph()

        inputanat = pe.Node(niu.IdentityInterface(fields=['data']), name='inputanat')
        inputanat.iterables = [('data', [list(s) for s in subjects['anat']])]
        dsanat = pe.Node(niu.Split(splits=[1, 1, 1, 1], squeeze=True),
                         name='ds_anat')

        # re_csv0 = pe.Node(niu.Function(
        #     input_names=['csv_file'], output_names=['out_file'], function=reorder_csv),
        #     name='reorder_anat')
        re_csv0 = pe.JoinNode(niu.Function(
            input_names=['csv_file'], output_names=['out_file'], function=reorder_csv),
                              joinsource='inputanat', joinfield='csv_file', name='reorder_anat')
        report0 = pe.Node(Report(qctype='anatomical', settings=settings), name='AnatomicalReport')
        report0.inputs.sub_list = subjects['anat']

        wf.connect([
            (inputanat, dsanat, [('data', 'inlist')]),
            (dsanat, anat_wf,   [('out1', 'inputnode.subject_id'),
                                 ('out2', 'inputnode.session_id'),
                                 ('out3', 'inputnode.scan_id'),
                                 ('out4', 'inputnode.anatomical_scan')]),
            (anat_wf, re_csv0,  [('outputnode.out_csv', 'csv_file')]),
            (re_csv0, outputnode, [('out_file', 'anat_csv')]),
            (re_csv0, report0,  [('out_file', 'in_csv')])
        ])

    if 'func' not in skip and subjects['func']:
        func_wf = fmri_qc_workflow(settings=settings)
        if settings.get('write_graph', False):
            func_wf.write_graph()

        inputfunc = pe.Node(niu.IdentityInterface(fields=['data']), name='inputfunc')
        inputfunc.iterables = [('data', [list(s) for s in subjects['func']])]
        dsfunc = pe.Node(niu.Split(splits=[1, 1, 1, 1], squeeze=True),
                         name='ds_func')
        re_csv1 = pe.JoinNode(niu.Function(
            input_names=['csv_file'], output_names=['out_file'], function=reorder_csv),
                              joinsource='inputfunc', joinfield='csv_file', name='reorder_func')
        report1 = pe.Node(Report(qctype='functional', settings=settings), name='FunctionalReport')
        report1.inputs.sub_list = subjects['func']

        wf.connect([
            (inputfunc, dsfunc, [('data', 'inlist')]),
            (dsfunc, func_wf,   [('out1', 'inputnode.subject_id'),
                                 ('out2', 'inputnode.session_id'),
                                 ('out3', 'inputnode.scan_id'),
                                 ('out4', 'inputnode.functional_scan')]),
            (func_wf, re_csv1,  [(('outputnode.out_csv', _back), 'csv_file')]),
            (re_csv1, outputnode, [('out_file', 'func_csv')]),
            (re_csv1, report1,  [('out_file', 'in_csv')])
        ])
    return wf
