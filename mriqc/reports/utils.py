#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# @Author: oesteban
# @Date:   2016-01-05 11:33:39
# @Email:  code@oscaresteban.es
# @Last modified by:   oesteban
# @Last Modified time: 2016-10-17 08:06:37
""" Helpers in report generation """
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from __future__ import unicode_literals
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


def plot_anat_mosaic_helper(in_file, subject_id, session_id,
                            run_id, out_name, bbox_mask_file=None,
                            title='T1w session: {session_id} run: {run_id}',
                            only_plot_noise=False):
    from mriqc.interfaces.viz_utils import plot_mosaic
    import os
    title = title.format(**{"session_id": session_id,
                          "run_id": run_id})
    fig = plot_mosaic(in_file, bbox_mask_file=bbox_mask_file, title=title,
                      only_plot_noise=only_plot_noise)
    fig.savefig(out_name, format=out_name.split('.')[-1], dpi=300)

    return os.path.abspath(out_name)

def iqms2html(indict):
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

    result_str = '<table id="iqms-table">\n'
    td = lambda v, cols: '<td colspan={1}>{0}</td>'.format(v, cols) if cols > 0 else '<td>{}</td>'.format(v)
    for line in sorted(out_qc):
        result_str += '<tr>'
        ncols = len(line)
        for i, col in enumerate(line):
            colspan = 0
            if (depth - ncols) > 0 and i == ncols - 2:
                colspan =  (depth - ncols) + 1
            result_str += td(col, colspan)

        result_str += '</tr>\n'

    result_str += '</table>\n'
    return result_str

