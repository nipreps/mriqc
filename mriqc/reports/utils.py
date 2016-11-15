#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# @Author: oesteban
# @Date:   2016-01-05 11:33:39
# @Email:  code@oscaresteban.es
# @Last modified by:   oesteban
# @Last Modified time: 2016-11-14 17:13:42
""" Helpers in report generation """
from __future__ import print_function, division, absolute_import, unicode_literals
import os.path as op
import re
import pandas as pd

def read_csv(in_csv):
    """ Read csv file, sort and drop duplicates """
    dframe = pd.read_csv(in_csv, dtype={'subject_id': str})
    try:
        dframe = dframe.sort_values(by=['subject_id', 'session_id', 'run_id'])
    except AttributeError:
        #pylint: disable=E1101
        dframe = dframe.sort(columns=['subject_id', 'session_id', 'run_id'])

    try:
        #pylint: disable=E1101
        dframe.drop_duplicates(['subject_id', 'session_id', 'run_id'], keep='last',
                               inplace=True)
    except TypeError:
        #pylint: disable=E1101
        dframe.drop_duplicates(['subject_id', 'session_id', 'run_id'], take_last=True,
                               inplace=True)
    #pylint: disable=E1101
    subject_list = sorted(pd.unique(dframe.subject_id.ravel()))
    return dframe, subject_list


def find_failed(dframe, sub_list):
    """ Identify failed subjects """
    sub_list = [(s[0], s[1], s[2]) for s in sub_list]
    success = [tuple(x) for x in dframe[['subject_id', 'session_id', 'run_id']].values]
    failed = list(set(sub_list) - set(success))
    return failed

def iqms2html(indict, table_id):
    if indict is None or not indict:
        return None

    tmp_qc = {}
    for k, value in sorted(list(indict.items())):
        if not isinstance(value, dict):
            tmp_qc[k] = value
        else:
            for subk, subval in sorted(list(value.items())):
                if not isinstance(subval, dict):
                    tmp_qc[','.join([k, subk])] = subval
                else:
                    for ssubk, ssubval in sorted(list(subval.items())):
                        tmp_qc[','.join([k, subk, ssubk])] = ssubval
    out_qc = []
    depth = 0
    for k, v in list(tmp_qc.items()):
        out_qc.append(k.split(',') + [v])
        if len(out_qc[-1]) > depth:
            depth = len(out_qc[-1])

    result_str = '<table id="%s">\n' % table_id
    td = lambda v, cols: '<td colspan={1}>{0}</td>'.format(v, cols) if cols > 0 else '<td>{}</td>'.format(v)
    for line in sorted(out_qc):
        result_str += '<tr>'
        ncols = len(line)
        for i, col in enumerate(line):
            colspan = 0
            if (depth - ncols) > 0 and i == ncols - 2:
                colspan = (depth - ncols) + 1
            result_str += td(col, colspan)

        result_str += '</tr>\n'

    result_str += '</table>\n'
    return result_str

def check_reports(dataset, settings, save_failed=True):

    supported_components = ['subject_id', 'session_id', 'task_id', 'run_id']
    expr = re.compile('^(?P<subject_id>sub-[a-zA-Z0-9]+)(_(?P<session_id>ses-[a-zA-Z0-9]+))?'
                      '(_(?P<task_id>task-[a-zA-Z0-9]+))?(_(?P<acq_id>acq-[a-zA-Z0-9]+))?'
                      '(_(?P<rec_id>rec-[a-zA-Z0-9]+))?(_(?P<run_id>run-[a-zA-Z0-9]+))?')

    reports_missed = False
    missing = {}
    for mod, files in list(dataset.items()):
        missing[mod] = []
        qctype = 'anatomical' if mod == 't1w' else 'functional'

        for fname in files:
            m = expr.search(op.basename(fname)).groupdict()
            components = [m.get(key) for key in supported_components if m.get(key)]
            components.insert(0, qctype)

            report_fname = op.join(
                settings['report_dir'], '_'.join(components) + '_report.html')

            if not op.isfile(report_fname):
                missing[mod].append(
                    {key: m.get(key) for key in supported_components if m.get(key)})

        mod_missing = missing[mod]
        if mod_missing:
            reports_missed = True

        if mod_missing and save_failed:
            out_file = op.join(settings['output_dir'], 'failed_%s.csv' % qctype)
            miss_cols = list(set(supported_components) & set(list(mod_missing[0].keys())))
            dframe = pd.DataFrame.from_dict(mod_missing).sort_values(
                by=miss_cols)
            dframe[miss_cols].to_csv(out_file, index=False)

    return reports_missed
