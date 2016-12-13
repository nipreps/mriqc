#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: oesteban
# @Date:   2015-11-19 16:44:27
# @Last Modified by:   oesteban
# @Last Modified time: 2016-12-12 16:50:17

"""
MRIQC Cross-validation

"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from scipy.stats import zscore
import pandas as pd
from builtins import str

from mriqc import logging
LOG = logging.getLogger('mriqc.classifier')

def read_dataset(feat_file, label_file):
    """ Reads in the features and labels """

    bids_comps = ['subject_id', 'session_id', 'task_id', 'run_id']

    x_df = pd.read_csv(feat_file, index_col=False,
        dtype={col: str for col in bids_comps})

    bids_comps_present = list(set(x_df.columns.ravel().tolist()) & set(bids_comps))
    x_df = x_df.sort_values(by=bids_comps_present)

    x_df['subject_id'] = x_df['subject_id'].map(lambda x: x.lstrip('sub-'))

    # Remove columns that are not IQMs
    feat_names = list(x_df._get_numeric_data().columns.ravel())

    for col in ['subject_id', 'session_id', 'run_id', 'qc_type']:
        try:
            feat_names.remove(col)
        except ValueError:
            pass

    for col in feat_names:
        if col.startswith(('size_', 'spacing_', 'Unnamed')):
            feat_names.remove(col)

    # Massage labels table to have the appropriate format
    y_df = pd.read_csv(label_file, index_col=False, dtype={'subject_id': object}).sort_values(
        by=['subject_id'])
    x_df['subject_id'] = x_df['subject_id'].map(lambda x: str(x))

    # Remove failed cases from Y, append new columns to X
    y_df = y_df[y_df['subject_id'].isin(list(x_df.subject_id.values.ravel()))]

    # Merge Y dataframe into X
    x_df = pd.merge(x_df, y_df, on='subject_id', how='left')

    return x_df, feat_names


def zscore_dataset(dataframe, excl_columns=None, by='site'):
    """ Returns a dataset zscored by the column given as argument """

    sites = list(dataframe[[by]].values.ravel())
    columns = list(dataframe._get_numeric_data().columns.ravel())

    if excl_columns is None:
        excl_columns = []

    for col in columns:
        if not np.isfinite(np.sum(dataframe[[col]].values.ravel())):
            excl_columns.append(col)

    if excl_columns:
        for col in excl_columns:
            try:
                columns.remove(col)
            except ValueError:
                pass

    zs_df = dataframe.copy()
    for site in sites:
        site_df = zs_df.loc[zs_df.site == site, columns]
        zscored = zscore(site_df, ddof=1, axis=0)
        for i, col in enumerate(columns):
            if not np.isnan(zscored[:, i]).any():
                zs_df.loc[zs_df.site == site, col] = zscored[:, i]

            if not np.isfinite(zs_df.loc[zs_df.site == site, col].sum()):
                LOG.warn('Sum of measure %s of %s\'s sample is infinite',
                         col, site)
                zs_df.drop(col, axis=1, inplace=True)
                break

    return zs_df
