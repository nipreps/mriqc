#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: oesteban
# @Date:   2015-11-19 16:44:27
# @Last Modified by:   oesteban
# @Last Modified time: 2016-05-26 16:56:00

"""
MRIQC Cross-validation

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import os.path as op
import simplejson as json
from argparse import ArgumentParser
from argparse import RawTextHelpFormatter

import numpy as np
from scipy.stats.mstats import zscore
import pandas as pd

from sklearn.cross_validation import LeaveOneLabelOut


from .estimator_helper import EstimatorSelectionHelper

DEFAULT_TEST_PARAMETERS = {
    'svc_linear': [{
        'C': [0.01, 0.1, 1, 10, 100, 100]
    }],
    'svc_rbf': [{
        'kernel': ['rbf'],
        'gamma': [1e-2, 1e-3, 1e-4],
        'C': [0.01, 0.1, 1, 10, 100]
    }],
    'rfc': [{
        'n_estimators': range(5, 20),
        'max_depth': [None] + range(5, 11),
        'min_samples_split': range(1, 5)
    }]
}

def main():
    """Entry point"""
    parser = ArgumentParser(description='MRI Quality Control',
                            formatter_class=RawTextHelpFormatter)

    g_input = parser.add_argument_group('Inputs')
    g_input.add_argument('-X', '--in-training', action='store',
                         required=True)
    g_input.add_argument('-y', '--in-training-labels', action='store',
                         required=True)
    g_input.add_argument('-N', '--site-normalization', action='store_true',
                         default=False)
    g_input.add_argument('-P', '--parameters', action='store')
    g_input.add_argument('-C', '--classifier', action='store', nargs='*',
                         choices=['svc_linear', 'svc_rbf', 'rfc', 'all'],
                         default=['svc_rbf'])
    g_input.add_argument(
        '-S', '--score-types', action='store', nargs='*',
        choices=['accuracy', 'adjusted_rand_score', 'average_precision', 'f1', 'f1_macro', 'f1_micro',
                 'f1_samples', 'f1_weighted', 'log_loss', 'mean_absolute_error', 'mean_squared_error',
                 'median_absolute_error', 'precision', 'precision_macro', 'precision_micro', 'precision_samples',
                 'precision_weighted', 'r2', 'recall', 'recall_macro', 'recall_micro', 'recall_samples',
                 'recall_weighted', 'roc_auc'],
        default=['f1_weighted', 'accuracy'])

    g_outputs = parser.add_argument_group('Outputs')
    g_outputs.add_argument('-o', '--out-csv', action='store',
                           help='output CSV file combining X and y')
    opts = parser.parse_args()

    with open(opts.in_training, 'r') as in_file_x:
        X_df = pd.read_csv(in_file_x).sort_values(by=['subject_id'])

    with open(opts.in_training_labels, 'r') as in_file_y:
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
    sites = list(y_df.site.values)
    X_df['rate'] = y_df.rate.values
    X_df['site'] = y_df.site.values

    if opts.site_normalization:
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

    X_df_test = X_df.sample(n=100, random_state=31051983)
    y_test = [int(y_i) for y_i in X_df_test.rate.values]
    X_test = [tuple(x) for x in X_df_test[colnames].values]

    if opts.out_csv is not None:
        with open(opts.out_csv, 'w') as outfile:
            X_df.to_csv(outfile, index=False)

    lolo_folds = LeaveOneLabelOut(sites)

    test_clfs = opts.classifier
    if 'all' in opts.classifier:
        test_clfs = ['svc_linear', 'svc_rbf', 'rfc']

    parameters = DEFAULT_TEST_PARAMETERS
    if opts.parameters is not None:
        with open(opts.parameters, 'r') as param_file:
            parameters = json.load(param_file)

    sample_x = [tuple(x) for x in X_df[colnames].values]
    labels_y = [int(y_i) for y_i in y_df.rate.values]

    helper = EstimatorSelectionHelper(
        test_clfs, parameters, scorings=opts.score_types)
    helper.fit(sample_x, labels_y, cv=lolo_folds, n_jobs=-1)
    helper.score_summary(out_file='results_cv.csv')


if __name__ == '__main__':
    main()
