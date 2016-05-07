#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# @Author: oesteban
# @Date:   2016-01-05 11:33:39
# @Email:  code@oscaresteban.es
# @Last modified by:   oesteban
# @Last Modified time: 2016-04-21 14:39:29
""" Helpers in report generation """

import numpy as np
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


def image_parameters(dframe):
    """ Generate formatted parameters for each subject, session and scan """
    newdf = dframe.copy()
    # Pack together subject session & scan as identifier
    newdf['id'] = [tuple(val[1]) for val in newdf[['subject_id', 'session_id', 'run_id']].iterrows()]

    # Format the size
    #pylint: disable=E1101
    newdf[['size_x', 'size_y', 'size_z']] = newdf[['size_x', 'size_y', 'size_z']].astype(np.uint16)
    formatter = lambda row: '%d &times; %d &times; %d' % (
        row['size_x'], row['size_y'], row['size_z'])
    newdf['size'] = newdf[['size_x', 'size_y', 'size_z']].apply(formatter, axis=1)

    # Format spacing
    newdf[['spacing_x', 'spacing_y', 'spacing_z']] = newdf[[
        'spacing_x', 'spacing_y', 'spacing_z']].astype(np.float32)  #pylint: disable=E1101
    newdf['spacing'] = zip(newdf.spacing_x, newdf.spacing_y, newdf.spacing_z)
    formatter = lambda row: '%.3f &times; %.3f &times; %.3f' % (
        row['spacing_x'], row['spacing_y'], row['spacing_z'])
    newdf['spacing'] = newdf[['spacing_x', 'spacing_y', 'spacing_z']].apply(formatter, axis=1)
    cols = ['size', 'spacing']

    if 'tr' in newdf.columns.ravel():
        cols.append('tr')
    if 'size_t' in newdf.columns.ravel():
        cols.append('size_t')

    return newdf.set_index('id')[cols].to_dict(orient='index')
