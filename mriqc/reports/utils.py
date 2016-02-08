#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# @Author: oesteban
# @Date:   2016-01-05 11:33:39
# @Email:  code@oscaresteban.es
# @Last modified by:   oesteban
# @Last Modified time: 2016-01-18 18:05:18
""" Helpers in report generation """

import pandas as pd


def read_csv(in_csv):
    """ Read csv file, sort and drop duplicates """
    dframe = pd.read_csv(in_csv, dtype={'subject': str})
    try:
        dframe = dframe.sort_values(by=['subject', 'session', 'scan'])
    except AttributeError:
        #pylint: disable=E1101
        dframe = dframe.sort(columns=['subject', 'session', 'scan'])

    try:
        #pylint: disable=E1101
        dframe.drop_duplicates(['subject', 'session', 'scan'], keep='last',
                               inplace=True)
    except TypeError:
        #pylint: disable=E1101
        dframe.drop_duplicates(['subject', 'session', 'scan'], take_last=True,
                               inplace=True)
    #pylint: disable=E1101
    subject_list = sorted(pd.unique(dframe.subject.ravel()))
    return dframe, subject_list


def find_failed(dframe, sub_list):
    """ Identify failed subjects """
    sub_list = [(s[0], s[1], s[2]) for s in sub_list]
    success = [tuple(x) for x in dframe[['subject', 'session', 'scan']].values]
    failed = list(set(sub_list) - set(success))
    return failed


def image_parameters(dframe):
    """ Generate formatted parameters for each subject, session and scan """
    newdf = dframe.copy()
    # Pack together subject session & scan as identifier
    newdf['id'] = zip(newdf.subject, newdf.session, newdf.scan)

    # Format the size
    newdf['size'] = zip(newdf.size_x, newdf.size_y, newdf.size_z)
    formatter = lambda x: '%d &times; %d &times; %d' % x
    newdf['size'] = newdf['size'].apply(formatter)

    # Format spacing
    newdf['spacing'] = zip(newdf.spacing_x, newdf.spacing_y, newdf.spacing_z)
    formatter = lambda x: '%f &times; %f &times; %f' % x
    newdf['spacing'] = newdf['spacing'].apply(formatter)
    cols = ['size', 'spacing']

    return newdf.set_index('id')[cols].to_dict(orient='index')
