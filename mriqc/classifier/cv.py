#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: oesteban
# @Date:   2015-11-19 16:44:27
# @Last Modified by:   oesteban
# @Last Modified time: 2016-05-20 10:22:52

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
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.cross_validation import LeaveOneLabelOut, permutation_test_score
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report, f1_score, accuracy_score

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
        default=['f1_weighted'])

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

    if 'svc_linear' in test_clfs:
        for stype in opts.score_types:
            sample_x = [tuple(x) for x in X_df[colnames].values]
            labels_y = [int(y_i) for y_i in y_df.rate.values]
            clf = GridSearchCV(
                svm.LinearSVC(C=1), convert([{'C': [0.01, 0.1, 1, 10, 100, 1000]}]),
                cv=lolo_folds, scoring=stype, n_jobs=-1)
            clf.fit(sample_x, labels_y)

            print("Best parameters set found on development set:")
            print()
            print(clf.best_params_)
            print()
            print("Grid %s scores on development set:" % stype)
            print()
            for params, mean_score, scores in clf.grid_scores_:
                print("%0.3f (+/-%0.03f) for %r"
                      % (mean_score, scores.std() * 2, params))
            print()
            print("Detailed classification report:")
            print()
            print("The model is trained on the full development set.")
            print("The scores are computed on the full evaluation set.")
            print()
            y_pred = clf.predict(X_test)
            print(classification_report(y_test, y_pred))
            print()
            print("Permutation test, p-values F1/Acc = %f / %f" % permutation_distribution(y_test, y_pred))

    if 'svc_rbf' in test_clfs:
    # Set the parameters by cross-validation
        if opts.parameters is not None:
            with open(opts.parameters, 'r') as paramfile:
                tuned_parameters = convert(json.load(paramfile))
        else:
            tuned_parameters = convert([
                {'kernel': ['rbf'], 'gamma': [1e-2, 1e-3, 1e-4], 'C': [0.1, 1, 10, 100, 1000]}
            ])
        for stype in opts.score_types:
            sample_x = [tuple(x) for x in X_df[colnames].values]
            labels_y = [int(y_i) for y_i in y_df.rate.values]
            clf = GridSearchCV(svm.SVC(C=1), tuned_parameters, cv=lolo_folds,
                               scoring=stype, n_jobs=-1)
            clf.fit(sample_x, labels_y)

            print("Best parameters set found on development set:")
            print()
            print(clf.best_params_)
            print()
            print("Grid %s scores on development set:" % stype)
            print()
            for params, mean_score, scores in clf.grid_scores_:
                print("%0.3f (+/-%0.03f) for %r"
                      % (mean_score, scores.std() * 2, params))
            print()
            print("Detailed classification report:")
            print()
            print("The model is trained on the full development set.")
            print("The scores are computed on the full evaluation set.")
            print()
            y_pred = clf.predict(X_test)
            print(classification_report(y_test, y_pred))
            print()
            print("Permutation test, p-values F1/Acc = %f / %f" % permutation_distribution(y_test, y_pred))


    if 'rfc' in test_clfs:
        # Set the parameters by cross-validation
        tuned_parameters = convert([
            {'n_estimators': range(5, 20),
             'max_depth': [None] + range(5, 11),
             'min_samples_split': range(1, 5)}
        ])
        for stype in opts.score_types:
            sample_x = [tuple(x) for x in X_df[colnames].values]
            labels_y = [int(y_i) for y_i in y_df.rate.values]
            clf = GridSearchCV(RFC(), tuned_parameters, cv=lolo_folds,
                               scoring=stype, n_jobs=-1)
            clf.fit(sample_x, labels_y)

            print("Best parameters set found on development set:")
            print()
            print(clf.best_params_)
            print()
            print("Grid %s scores on development set:" % stype)
            print()
            for params, mean_score, scores in clf.grid_scores_:
                print("%0.3f (+/-%0.03f) for %r"
                      % (mean_score, scores.std() * 2, params))
            print()
            print("Detailed classification report:")
            print()
            print("The model is trained on the full development set.")
            print("The scores are computed on the full evaluation set.")
            print()
            y_pred = clf.predict(X_test)
            print(classification_report(y_test, y_pred))
            print()
            print("Permutation test, p-values F1/Acc = %f / %f" % permutation_distribution(y_test, y_pred))


def permutation_distribution(y_true, y_pred, n_permutations=5e4):
    """ Compute the distribution of permutations """
    # Save actual f1_score in front
    random_f1 = []
    random_acc = []
    for i in range(int(n_permutations)):
        y_sh = np.random.permutation(y_true)
        random_f1.append(f1_score(y_sh, y_pred))
        random_acc.append(accuracy_score(y_sh, y_pred))

    random_f1 = np.array(random_f1)
    random_acc = np.array(random_acc)

    pval_f1 = ((len(random_f1[random_f1 > f1_score(y_true, y_pred)]) + 1) /
               float(n_permutations + 1))
    pval_acc = ((len(random_acc[random_acc > accuracy_score(y_true, y_pred)]) + 1) /
                float(n_permutations + 1))
    return pval_f1, pval_acc


def convert(data):
    from collections import Mapping, Iterable
    if isinstance(data, basestring):
        return str(data)
    elif isinstance(data, Mapping):
        return dict(map(convert, data.iteritems()))
    elif isinstance(data, Iterable):
        return type(data)(map(convert, data))
    else:
        return data


if __name__ == '__main__':
    main()
