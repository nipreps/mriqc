#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: oesteban
# @Date:   2015-11-19 16:44:27
# @Last Modified by:   oesteban
# @Last Modified time: 2017-01-24 11:48:25

"""
mriqc_fit: data handling module

"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import pandas as pd
from builtins import str

from mriqc import logging
LOG = logging.getLogger('mriqc.classifier')

def read_dataset(feat_file, label_file, rate_label='rate', merged_name=None,
                 binarize=True):
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
    y_df = pd.read_csv(
        label_file, index_col=False, dtype={'subject_id': object}).sort_values(by=['subject_id'])
    y_df['subject_id'] = y_df['subject_id'].map(lambda x: x.lstrip('sub-'))
    x_df['subject_id'] = x_df['subject_id'].map(lambda x: str(x))

    # Remove failed cases from Y, append new columns to X
    y_df = y_df[y_df['subject_id'].isin(list(x_df.subject_id.values.ravel()))]

    # Merge Y dataframe into X
    x_df = pd.merge(x_df, y_df, on='subject_id', how='left')

    if merged_name is not None:
        x_df.to_csv(merged_name, index=False)

    # Drop samples with invalid rating
    nan_labels = x_df[x_df[rate_label].isnull()].index.ravel().tolist()
    if nan_labels:
        LOG.info('Dropping %d samples for having non-numerical '
                 'labels', len(nan_labels))
        x_df = x_df.drop(nan_labels)

    # Convert string labels to ints
    try:
        x_df.loc[x_df[rate_label].str.contains('fail', case=False, na=False), rate_label] = -1
        x_df.loc[x_df[rate_label].str.contains('exclude', case=False, na=False), rate_label] = -1
        x_df.loc[x_df[rate_label].str.contains('maybe', case=False, na=False), rate_label] = 0
        x_df.loc[x_df[rate_label].str.contains('may be', case=False, na=False), rate_label] = 0
        x_df.loc[x_df[rate_label].str.contains('ok', case=False, na=False), rate_label] = 1
        x_df.loc[x_df[rate_label].str.contains('good', case=False, na=False), rate_label] = 1
    except AttributeError:
        pass

    x_df[[rate_label]] = x_df[[rate_label]].apply(pd.to_numeric, errors='raise')

    if binarize:
        x_df.loc[x_df[rate_label] >= 0, rate_label] = 0
        x_df.loc[x_df[rate_label] < 0, rate_label] = 1

    # Print out some info
    nsamples = len(x_df)
    LOG.info('Created dataset X="%s", Y="%s" (N=%d valid samples)',
             feat_file, label_file, nsamples)

    nfails = int(x_df[rate_label].sum())
    LOG.info('Ratings distribution: "fail"=%d / "ok"=%d (%f%% failed)',
             nfails, nsamples - nfails, nfails * 100 / nsamples)

    return x_df, feat_names


def zscore_dataset(dataframe, excl_columns=None, by='site',
                   njobs=-1):
    """ Returns a dataset zscored by the column given as argument """
    from multiprocessing import Pool, cpu_count

    LOG.info('z-scoring dataset ...')

    if njobs <= 0:
        njobs = cpu_count()

    sites = list(set(dataframe[[by]].values.ravel().tolist()))
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

    pool = Pool(njobs)
    args = [(zs_df, columns, s) for s in sites]
    results = pool.map(zscore_site, args)
    for site, res in zip(sites, results):
        zs_df.loc[zs_df.site == site, columns] = res

    zs_df.replace([np.inf, -np.inf], np.nan)
    nan_columns = zs_df.columns[zs_df.isnull().any()].tolist()

    if nan_columns:
        LOG.warn('Columns %s contain NaNs after z-scoring.', ", ".join(nan_columns))
        zs_df[nan_columns] = dataframe[nan_columns].values

    return zs_df

def zscore_site(args):
    """ z-scores only one site """
    from scipy.stats import zscore
    dataframe, columns, site = args
    return zscore(dataframe.loc[dataframe.site == site, columns].values,
                  ddof=1, axis=0)
