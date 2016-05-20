#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# @Author: oesteban
# @Date:   2016-01-05 11:33:39
# @Email:  code@oscaresteban.es
# @Last modified by:   oesteban
# @Last Modified time: 2016-05-20 09:10:54
""" Helpers in report generation """
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
