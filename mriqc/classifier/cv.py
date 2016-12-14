#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: oesteban
# @Date:   2015-11-19 16:44:27
# @Last Modified by:   oesteban
# @Last Modified time: 2016-12-13 17:38:45

"""
MRIQC Cross-validation

"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import pandas as pd
import simplejson as json

from pprint import pformat as pf
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.metrics import classification_report, f1_score, accuracy_score

from builtins import object
from .data import read_dataset, zscore_dataset
from mriqc import __version__, logging

try:
    from sklearn.model_selection import (LeaveOneGroupOut, StratifiedKFold, GridSearchCV,
                                         permutation_test_score, PredefinedSplit, cross_val_score)
except ImportError:
    from sklearn.cross_validation import (
        permutation_test_score, StratifiedKFold,
        LeaveOneLabelOut as LeaveOneGroupOut,
        PredefinedSplit, cross_val_score)
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

        self._models = {}
        self.Xtest = None
        self.n_jobs = n_jobs
        self._rate_column = 'rate'
        LOG.info('Created CV object for dataset "%s" with labels "%s"', X, Y)

        nan_labels = self.X[np.isnan(self.X[self._rate_column])].index.ravel().tolist()
        if nan_labels:
            LOG.info('Dropping %d samples for having non-numerical '
                     'labels', len(nan_labels))
            self.X = self.X.drop(nan_labels)

        self.sites = list(set(self.X.site.values.ravel()))
        self.Xzscored = zscore_dataset(
            self.X, excl_columns=[self._rate_column, 'size_x', 'size_y', 'size_z',
                                  'spacing_x', 'spacing_y', 'spacing_z'])


    @property
    def rate_column(self):
        return self._rate_column

    def create_test_split(self, split_type='sample', **sample_args):
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

        self.Xtest_zscored = self.Xzscored.take(self.Xtest.index)
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

    def fit(self, folds=None):

        cv_params = {
            'n_jobs': self.n_jobs
        }

        folds_groups = None
        if folds is not None and folds.get('type', '') == 'kfold':
            nsplits = folds.get('n_splits', 6)
            cv_params['cv'] = StratifiedKFold(n_splits=nsplits, shuffle=True)
        else:
            LOG.info('No folds provided for CV, using default leave-one-site-out')
            folds_groups = list(self.X.site.values.ravel())
            cv_params['cv'] = LeaveOneGroupOut()

        for clf_type, _ in list(self.param.items()):
            self._models[clf_type] = []

        LOG.info('Starting inner cross-validation loop. N=%d', len(self.X))
        for dozs in [False, True]:
            X = self.Xzscored.copy() if dozs else self.X.copy()
            sample_x, labels_y = self._generate_sample(X)
            for clf_type, clf_params in list(self.param.items()):
                for stype in self.scores:
                    cv_params['scoring'] = stype
                    LOG.info('CV loop for %s, optimizing for %s, and %s zscoring',
                             clf_type, stype, 'with' if dozs else 'without')
                    clf = GridSearchCV(_clf_build(clf_type), clf_params,
                                       **cv_params)

                    clf.fit(sample_x, labels_y, groups=folds_groups)

                    thismodel = {
                        'scoring': stype,
                        'zscored': dozs,
                        'best_params': clf.best_params_,
                        'best_score': clf.best_score_,
                        'grid_scores': clf.cv_results_
                    }

                    LOG.info('Model selection finished. Running permutation test.')

                    outer_cv = StratifiedKFold(n_splits=5, shuffle=True)
                    LOG.info('Evaluating best classifier')
                    score, permutation_scores, pvalue = permutation_test_score(
                        clf, sample_x, labels_y, scoring=stype, cv=outer_cv, n_permutations=100)
                    LOG.info('Classification score %s (p-value=%s)', score, pvalue)

                    thismodel['classification_score'] = score
                    thismodel['permutation_scores'] = permutation_scores
                    thismodel['pvalue'] = pvalue
                    self._models[clf_type].append(thismodel)


    def get_best(self, scoring=None):
        if scoring is None:
            scoring = self.scores[0]

        if scoring not in self.scores:
            raise RuntimeError('CV did not compute any loop for '
                               '"%s" scoring.' % scoring)

        best_score = -1
        best_model = None
        for clf_type, clf_vals in list(self._models.items()):
            for vals in clf_vals:
                if vals['scoring'] == scoring and vals['classification_score'] > best_score:
                    best_model = vals
                    best_model['clf_type'] = clf_type
        return best_model

    def outer_loop(self):
        if self.Xtest is None:
            LOG.error('Test dataset is not set')

        best_model = self.get_best()

        LOG.info('Buiding best classifier (%s), with params:\n%s',
                 best_model['clf_type'], best_model['best_params'])
        clf = _clf_build(best_model['clf_type'])
        clf.set_params(**best_model['best_params'])

        X = self.Xzscored.copy() if best_model['zscored'] else self.X.copy()
        Xtest = self.Xtest_zscored.copy() if best_model['zscored'] else self.Xtest.copy()
        x, y = self._generate_sample(pd.concat([X, Xtest]))

        # test_fold = np.array([-1] * len(X) + [0] * len(Xtest))
        ps = StratifiedKFold(n_splits=5)
        LOG.info('Evaluating best classifier')
        score, permutation_scores, pvalue = permutation_test_score(
            clf, x, y, scoring="accuracy", cv=ps, n_permutations=10000)
        LOG.info('Classification score %s (p-value=%s)', score, pvalue)


    def _generate_sample(self, X):
        sample_x = np.array([tuple(x) for x in X[self.ftnames].values])
        labels_y = X[[self._rate_column]].values.ravel()
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
