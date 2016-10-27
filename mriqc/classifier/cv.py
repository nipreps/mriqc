#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: oesteban
# @Date:   2015-11-19 16:44:27
# @Last Modified by:   oesteban
# @Last Modified time: 2016-10-27 09:45:20

"""
MRIQC Cross-validation

"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import pandas as pd
from pprint import pformat as pf
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.cross_validation import permutation_test_score
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report, f1_score, accuracy_score

from builtins import object
from .data import read_dataset, zscore_dataset
from mriqc import __version__, logging

try:
    from sklearn.cross_validation import LeaveOneGroupOut
except ImportError:
    from sklearn.cross_validation import LeaveOneLabelOut as LeaveOneGroupOut

LOG = logging.getLogger('mriqc.classifier')

DEFAULT_TEST_PARAMETERS = {
    b'svc_linear': [{
        b'C': [0.01, 0.1, 1, 10, 100, 100]
    }],
    b'svc_rbf': [{
        b'kernel': [b'rbf'],
        b'gamma': [1e-2, 1e-3, 1e-4],
        b'C': [0.01, 0.1, 1, 10, 100]
    }],
    b'rfc': [{
        b'n_estimators': range(5, 20),
        b'max_depth': [None] + range(5, 11),
        b'min_samples_split': range(1, 5)
    }]
}

DEFAULT_TEST_PARAMETERS = {
    'svc_linear': [{
        'C': [0.001, 0.01, 0.1, 1, 10, 100, 100]
    }],
}


class CVHelper(object):

    def __init__(self, X, Y, scores=None, param=None, lo_label='site',
                 n_jobs=-1):
        self.X, self.ftnames = read_dataset(X, Y)
        self.lo_labels = list(set(self.X[[
            lo_label]].values.ravel().tolist()))

        self.scores = ['f1', 'accuracy']
        if scores is not None:
            self.scores = scores

        self.param = DEFAULT_TEST_PARAMETERS.copy()

        if param is not None:
            self.param = param

        self._models = []
        self.Xtest = None
        self.n_jobs = n_jobs
        LOG.info('Created CV object for dataset "%s" with labels "%s"', X, Y)


    def set_heldout_dataset(self, X, Y):
        self.Xtest, ftnames = read_dataset(X, Y)
        if set(self.ftnames) - set(ftnames):
            raise RuntimeError('Some features are missing in the held-out dataset')

    def _sample_from_test(self):
        self.Xtest = self.X.sample(n=100, random_state=31051983)


    def inner_loop(self, folds=None):
        if folds is None:
            folds = LeaveOneGroupOut(list(self.X.site.values.ravel()))
            LOG.info('No folds provided for CV, using default leave-one-site-out')


        LOG.info('Starting inner cross-validation loop')
        for dozs in [False, True]:

            X = self.X.copy()

            if dozs:
                X = zscore_dataset(X, excl_columns=['rate', 'size_x', 'size_y', 'size_z',
                                                    'spacing_x', 'spacing_y', 'spacing_z'])

            sample_x = [tuple(x) for x in X[self.ftnames].values]
            labels_y = list(X.rate.astype(np.uint8).values.ravel())

            for clf_type, clf_params in list(self.param.items()):
                for stype in self.scores:
                    LOG.info('CV loop for %s, optimizing for %s, and %s zscoring',
                             clf_type, stype, 'with' if dozs else 'without')
                    innercv = GridSearchCV(_clf_build(clf_type),
                                       clf_params,
                                       scoring=stype,
                                       cv=folds,
                                       n_jobs=self.n_jobs,
                                       refit=False)

                    innercv.fit(sample_x, labels_y)

                    self._models.append({
                        'clf_type': clf_type,
                        'scoring': stype,
                        'zscored': dozs,
                        'best_params': innercv.best_params_,
                        'best_score': innercv.best_score_,
                        'grid_scores': innercv.grid_scores_
                    })

                    LOG.info('CV loop finished: \n%s', pf(self._models[-1], indent=2))

    def get_best(self, scoring=None, refit=True):
        if scoring is None:
            scoring = self.scores[0]

        if scoring not in self.scores:
            raise RuntimeError('CV did not compute any loop for '
                               '"%s" scoring.' % scoring)

        best_score = -1
        best_model = None
        for model in self._models:
            if model['scoring'] == scoring and model['best_score'] > best_score:
                best_model = model
                best_score = model['best_score']

        if refit:
            raise NotImplementedError

        return best_model


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
