#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: oesteban
# @Date:   2015-11-19 16:44:27
# @Last Modified by:   oesteban
# @Last Modified time: 2016-12-12 16:51:47

"""
MRIQC Cross-validation

"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import pandas as pd
from pprint import pformat as pf
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.metrics import classification_report, f1_score, accuracy_score

from builtins import object
from .data import read_dataset, zscore_dataset
from mriqc import __version__, logging

try:
    from sklearn.model_selection import (LeaveOneGroupOut, KFold, GridSearchCV,
                                         permutation_test_score)
except ImportError:
    from sklearn.cross_validation import (
        permutation_test_score, KFold,
        LeaveOneLabelOut as LeaveOneGroupOut)
    from sklearn.grid_search import GridSearchCV

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
        self._rate_column = 'rate'
        LOG.info('Created CV object for dataset "%s" with labels "%s"', X, Y)
        self.sites = list(set(self.X.site.values.ravel()))
        self.Xzscored = zscore_dataset(
            X, excl_columns=[self._rate_column, 'size_x', 'size_y', 'size_z',
                             'spacing_x', 'spacing_y', 'spacing_z'])


    @property
    def rate_column(self):
        return self._rate_column

    @rate_column.setter
    def rate_column(self, value):
        self._rate_column = value

    def create_test_split(self, split_type='sample', rate_column=None,
                          **sample_args):

        if rate_column is not None:
            self.rate_column = rate_column
            nan_labels = self.X[np.isnan(self.X[rate_column])].index.ravel().tolist()
            if nan_labels:
                LOG.info('Dropping %d samples for having non-numerical '
                         'labels', len(nan_labels))
                self.X = self.X.drop(nan_labels)

        if split_type == 'sample':
            self.Xtest = self.X.sample(**sample_args)
        elif split_type == 'site':
            newentries = []
            for site in self.sites:
                sitedf = self.X[self.X.site == site]
                newentries.append(sitedf.sample(**sample_args))
            self.Xtest = pd.concat(newentries)
        else:
            raise RuntimeError('Unknown split_type (%s).' % split_type)

        self.Xtest_zscored = self.Xzscored[self.Xtest.index]
        self.Xzscored = self.Xzscored.drop(self.Xtest.index)
        self.X = self.X.drop(self.Xtest.index)
        LOG.info('Created a random split of the data, the training set has '
                 '%d and the evaluation set %d.', len(self.X), len(self.Xtest))

    def set_heldout_dataset(self, X, Y):
        self.Xtest, ftnames = read_dataset(X, Y)
        if set(self.ftnames) - set(ftnames):
            raise RuntimeError('Some features are missing in the held-out dataset')

    def _sample_from_test(self, n=100):
        self.Xtest = self.X.sample(n=n, random_state=31051852)

    def to_csv(self, out_file, output_set='training'):
        if output_set == 'training':
            self.X.to_csv(out_file, index=False)
        elif output_set == 'evaluation':
            self.Xtest.to_csv(out_file, index=False)

    def inner_loop(self, folds=None):

        cv_params = {
            'n_jobs': self.n_jobs,
            'refit': False
        }

        folds_groups = None
        if folds is not None and folds.get('type', '') == 'kfold':
            nsplits = folds.get('n_splits', 10)
            cv_params['cv'] = KFold(n_splits=nsplits)
        else:
            LOG.info('No folds provided for CV, using default leave-one-site-out')
            folds_groups = list(self.X.site.values.ravel())
            cv_params['cv'] = LeaveOneGroupOut()

        LOG.info('Starting inner cross-validation loop')
        for dozs in [False, True]:
            X = self.Xzscored.copy() if dozs else self.X.copy()

            sample_x = [tuple(x) for x in X[self.ftnames].values]
            labels_y = X[[self._rate_column]].values.ravel().tolist()

            for clf_type, clf_params in list(self.param.items()):
                for stype in self.scores:
                    cv_params['scoring'] = stype
                    LOG.info('CV loop for %s, optimizing for %s, and %s zscoring',
                             clf_type, stype, 'with' if dozs else 'without')
                    innercv = GridSearchCV(_clf_build(clf_type), clf_params,
                                           **cv_params)

                    innercv.fit(sample_x, labels_y, groups=folds_groups)

                    self._models.append({
                        'clf_type': clf_type,
                        'scoring': stype,
                        'zscored': dozs,
                        'best_params': innercv.best_params_,
                        'best_score': innercv.best_score_,
                        'grid_scores': innercv.cv_results_
                    })

                    LOG.info('CV loop finished: \n%s', pf(self._models[-1], indent=2))

    def get_best(self, scoring=None):
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

        return best_model

    def evaluate_loop(self):

        if self.Xtest is None:
            LOG.error('Test dataset is not set')

        best_model = self.get_best()

        clf = _clf_build(best_model['clf_type'])
        clf.set_params(**best_model['best_params'])

        X = self.Xzscored.copy() if best_model['zscored'] else self.X.copy()
        clf.fit(_generate_sample(X))

        Xtest = self.Xtest_zscored.copy() if best_model['zscored'] else self.Xtest.copy()
        clf.score(_generate_sample(Xtest))

    def _generate_sample(self, X):
        sample_x = [tuple(x) for x in X[self.ftnames].values]
        labels_y = X[[self._rate_column]].values.ravel().tolist()
        return sample_x, labels_y


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
