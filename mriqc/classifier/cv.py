#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: oesteban
# @Date:   2015-11-19 16:44:27
# @Last Modified by:   oesteban
# @Last Modified time: 2016-05-13 11:07:01

"""
MRIQC Cross-validation

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import os.path as op
from argparse import ArgumentParser
from argparse import RawTextHelpFormatter

import pandas as pd
from sklearn import svm
from sklearn.cross_validation import LeaveOneLabelOut
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report

def main():
    """Entry point"""
    parser = ArgumentParser(description='MRI Quality Control',
                            formatter_class=RawTextHelpFormatter)

    g_input = parser.add_argument_group('Inputs')
    g_input.add_argument('-X', '--in-training', action='store',
                         required=True)
    g_input.add_argument('-y', '--in-training-labels', action='store',
                         required=True)

    g_outputs = parser.add_argument_group('Outputs')
    g_outputs.add_argument('-o', '--out-csv', action='store',
                           help='output CSV file combining X and y')
    opts = parser.parse_args()

    with open(opts.in_training, 'r') as in_file_x:
        X_df = pd.read_csv(in_file_x).sort_values(by=['subject_id'])

    with open(opts.in_training_labels, 'r') as in_file_y:
        y_df = pd.read_csv(in_file_y).sort_values(by=['subject_id'])

    # Remove columns that are not IQMs
    columns = list(X_df.columns.ravel())
    columns.remove('subject_id')
    columns.remove('session_id')
    columns.remove('run_id')

    # Remove failed cases from Y, append new columns to X
    y_df = y_df[y_df['subject_id'].isin(X_df.subject_id)]
    sites = list(y_df.site.values)
    X_df['rate'] = y_df.rate.values
    X_df['site'] = y_df.site.values

    if opts.out_csv is not None:
        with open(opts.out_csv, 'w') as outfile:
            X_df.to_csv(outfile, index=False)

    lolo_folds = LeaveOneLabelOut(sites)

    # Set the parameters by cross-validation
    tuned_parameters = [
        convert({'kernel': ['rbf'], 'gamma': [1e-3, 1e-4], 'C': [1, 10, 100, 1000]}),
        # convert({'kernel': ['linear'], 'C': [1, 10, 100, 1000]})
    ]

    scores = ['precision', 'recall']

    for score in scores:
        for train, test in lolo_folds:
            # Generate data splits
            X_train = [tuple(x) for x in X_df.ix[train, columns].values]
            y_train = [int(y_i) for y_i in y_df.iloc[train].rate.values]
            X_test = [tuple(x) for x in X_df.ix[test, columns].values]
            y_test = [int(y_i) for y_i in y_df.iloc[test].rate.values]

            clf = GridSearchCV(svm.SVC(C=1), tuned_parameters,
                               scoring='%s_weighted' % score)
            clf.fit(X_train, y_train)

            print("Best parameters set found on development set:")
            print()
            print(clf.best_params_)
            print()
            print("Grid scores on development set:")
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
            y_true, y_pred = y_test, clf.predict(X_test)
            print(classification_report(y_true, y_pred))
            print()


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
