#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: oesteban
# @Date:   2015-11-19 16:44:27
# @Last Modified by:   oesteban
# @Last Modified time: 2016-10-26 11:25:24

"""
MRIQC Cross-validation

"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.cross_validation import LeaveOneLabelOut, permutation_test_score
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report, f1_score, accuracy_score

from builtin import object
from .data import read_dataset


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

class CVHelper(object):

    def __init__(self, X, Y, scores=None, param=None, lo_label='site',
                 n_jobs=-1):
        self.X, self.ftnames = read_dataset(X, Y)
        self.lo_labels = list(set(self.X[[
            lo_label]].values.ravel().tolist()))

        self.scores = ['f1_weighted', 'accuracy']
        if scores is not None:
            self.scores = scores

        self.param = DEFAULT_TEST_PARAMETERS.copy()

        if param is not None:
            self.param = param

        self.Xtest = None
        self.n_jobs = n_jobs


    def set_heldout_dataset(self, X, Y):
        self.Xtest, ftnames = read_dataset(X, Y)
        if set(self.ftnames) - set(ftnames):
            raise RuntimeError('Some features are missing in the held-out dataset')

    def _sample_from_test(self):
        self.Xtest = self.X.sample(n=100, random_state=31051983)


    def inner_loop(self, folds=None):
        if folds is None:
            folds = LeaveOneLabelOut(self.lo_labels)

        sample_x = [tuple(x) for x in self.X[self.ftnames].values]
        labels_y = [int(y_i) for y_i in self.X.rate.values]

        for clf_type, clf_params in list(self.params.items()):
            for stype in self.scores:
                clf = GridSearchCV(_clf_build(clf_type),
                                   clf_params,
                                   scorings=stype,
                                   cv=folds,
                                   n_jobs=self.n_jobs)
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


def _clf_build(clf_type):
    if clf_type == 'svc_linear':
        return svm.LinearSVC(C=1)
    elif clf_type == 'svc_rbf':
        return svm.SVC(C=1)
    elif clf_type == 'rfc':
        return RFC()


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


# def convert(data):
#     from collections import Mapping, Iterable
#     if isinstance(data, basestring):
#         return str(data)
#     elif isinstance(data, Mapping):
#         return dict(map(convert, data.iteritems()))
#     elif isinstance(data, Iterable):
#         return type(data)(map(convert, data))
#     else:
#         return data




        #         print()
        #         print("Detailed classification report:")
        #         print()
        #         print("The model is trained on the full development set.")
        #         print("The scores are computed on the full evaluation set.")
        #         print()
        #         y_pred = clf.predict(X_test)
        #         print(classification_report(y_test, y_pred))
        #         print()
        #         print("Permutation test, p-values F1/Acc = %f / %f" % permutation_distribution(y_test, y_pred))

        # if 'svc_rbf' in test_clfs:
        # # Set the parameters by cross-validation
        #     if opts.parameters is not None:
        #         with open(opts.parameters, 'r') as paramfile:
        #             tuned_parameters = convert(json.load(paramfile))
        #     else:
        #         tuned_parameters = convert([
        #             {'kernel': ['rbf'], 'gamma': [1e-2, 1e-3, 1e-4], 'C': [0.1, 1, 10, 100, 1000]}
        #         ])
        #     for stype in opts.score_types:
        #         sample_x = [tuple(x) for x in X_df[colnames].values]
        #         labels_y = [int(y_i) for y_i in y_df.rate.values]
        #         clf = GridSearchCV(svm.SVC(C=1), tuned_parameters, cv=lolo_folds,
        #                            scoring=stype, n_jobs=-1)
        #         clf.fit(sample_x, labels_y)

        #         print("Best parameters set found on development set:")
        #         print()
        #         print(clf.best_params_)
        #         print()
        #         print("Grid %s scores on development set:" % stype)
        #         print()
        #         for params, mean_score, scores in clf.grid_scores_:
        #             print("%0.3f (+/-%0.03f) for %r"
        #                   % (mean_score, scores.std() * 2, params))
        #         print()
        #         print("Detailed classification report:")

        #         print()
        #         print("The model is trained on the full development set.")
        #         print("The scores are computed on the full evaluation set.")
        #         print()
        #         y_pred = clf.predict(X_test)
        #         print(classification_report(y_test, y_pred))
        #         print()
        #         print("Permutation test, p-values F1/Acc = %f / %f" % permutation_distribution(y_test, y_pred))


        # if 'rfc' in test_clfs:
        #     # Set the parameters by cross-validation
        #     tuned_parameters = convert([
        #         {'n_estimators': range(5, 20),
        #          'max_depth': [None] + range(5, 11),
        #          'min_samples_split': range(1, 5)}
        #     ])
        #     for stype in opts.score_types:
        #         sample_x = [tuple(x) for x in X_df[colnames].values]
        #         labels_y = [int(y_i) for y_i in y_df.rate.values]
        #         clf = GridSearchCV(RFC(), tuned_parameters, cv=lolo_folds,
        #                            scoring=stype, n_jobs=-1)
        #         clf.fit(sample_x, labels_y)

        #         print("Best parameters set found on development set:")
        #         print()
        #         print(clf.best_params_)
        #         print()
        #         print("Grid %s scores on development set:" % stype)
        #         print()
        #         for params, mean_score, scores in clf.grid_scores_:
        #             print("%0.3f (+/-%0.03f) for %r"
        #                   % (mean_score, scores.std() * 2, params))
        #         print()
        #         print("Detailed classification report:")
        #         print()
        #         print("The model is trained on the full development set.")
        #         print("The scores are computed on the full evaluation set.")
        #         print()
        #         y_pred = clf.predict(X_test)
        #         print(classification_report(y_test, y_pred))
        #         print()
        #         print("Permutation test, p-values F1/Acc = %f / %f" % permutation_distribution(y_test, y_pred))
