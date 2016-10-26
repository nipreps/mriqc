#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: oesteban
# @Date:   2015-11-19 16:44:27
# @Last Modified by:   oesteban
# @Last Modified time: 2016-05-26 16:56:00

"""
MRIQC Cross-validation

"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from scipy.stats.mstats import zscore
import pandas as pd

from io import open

def read_dataset(feat_file, label_file):
    """ Reads in the features and labels """

    with open(feat_file, 'r') as in_file_x:
        X_df = pd.read_csv(in_file_x).sort_values(by=['subject_id'])

    with open(label_file, 'r') as in_file_y:
        y_df = pd.read_csv(in_file_y).sort_values(by=['subject_id'])

    # Remove columns that are not IQMs
    colnames = list(X_df.columns.ravel())
    colnames.remove('subject_id')
    colnames.remove('session_id')
    colnames.remove('run_id')
    for axis in ['x', 'y', 'z']:
        colnames.remove('size_' + axis)
        colnames.remove('spacing_' + axis)

    # Remove failed cases from Y, append new columns to X
    y_df = y_df[y_df['subject_id'].isin(X_df.subject_id)]
    X_df['rate'] = y_df.rate.values
    X_df['site'] = y_df.site.values

    return X_df, y_df, feat_names


def zscore_dataset(features, column):
    """ Returns a dataset zscored by the column given as argument """

    sites = list(y_df.site.values)
    for site in sites:
        for col in colnames:
            X_df.loc[X_df.site == site, col] = zscore(
                X_df.loc[X_df.site == site, col].values.tolist(), ddof=1)
    # Z-Scoring voxnum and voxsize will fail for sites with homogeneous
    # acquisition matrix or resolution across all subjects (sigma is 0).
    X_df['voxnum'] = np.array(
        X_df.size_x.values * X_df.size_y.values * X_df.size_z.values, dtype=np.uint32)
    X_df['voxsize'] = np.array(
        X_df.spacing_x.values * X_df.spacing_y.values * X_df.spacing_z.values, dtype=np.float32)

    return feat_zscored
