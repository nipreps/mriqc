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

    def __init__(self, X, Y, scores=None, param=None, n_jobs=-1, n_perm=5000,
                 site_label='site', rate_label='rate', cv_test=False):

        # Initialize some values
        self.scores = ['accuracy']
        if scores is not None:
            self.scores = scores

        self.param = DEFAULT_TEST_PARAMETERS.copy()
        if param is not None:
            self.param = param

        self.Xtest = None
        self.n_jobs = n_jobs
        self._rate_column = rate_label
        self._site_column = site_label

        self.X, self.ftnames = read_dataset(X, Y, rate_label=rate_label)
        self.sites = list(set(self.X[site_label].values.ravel()))
        self.Xzscored = zscore_dataset(
            self.X, excl_columns=[rate_label, 'size_x', 'size_y', 'size_z',
                                  'spacing_x', 'spacing_y', 'spacing_z'])

        self._models = {}
        self._best_clf = {}
        self._best_model = {}
        self.n_perm = n_perm
        self._cv_test = cv_test


    @property
    def rate_column(self):
        return self._rate_column

    @property
    def best_clf(self):
        return self._best_clf

    @property
    def best_model(self):
        return self._best_model

    @property
    def cv_test(self):
        return self._cv_test

    @cv_test.setter
    def cv_test(self, value):
        self._cv_test = True if value else False


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
            outer_nsplits = nsplits - 1
            outer_cv = StratifiedKFold(n_splits=outer_nsplits, shuffle=True)
            LOG.info('Cross validation: using StratifiedKFold, inner loop is %d-fold and '
                     ' outer loop is %d-fold', nsplits, outer_nsplits)
        else:
            folds_groups = list(self.X[[self._site_column]].values.ravel())
            cv_params['cv'] = LeaveOneGroupOut()
            outer_cv = LeaveOneGroupOut()
            LOG.info('Cross validation: using default leave-one-site-out')

        for clf_type, _ in list(self.param.items()):
            self._models[clf_type] = []

        best_model = 0.0
        for dozs in [False, True]:
            X = self.Xzscored.copy() if dozs else self.X.copy()
            sample_x, labels_y = self._generate_sample(X)
            for clf_type, clf_params in list(self.param.items()):
                for stype in self.scores:
                    thismodel = {'scoring': stype, 'zscored': dozs, 'clf_type': clf_type}
                    cv_params['scoring'] = stype
                    LOG.info('CV loop for %s, optimizing for %s, and %s zscoring',
                             clf_type, stype, 'with' if dozs else 'without')
                    clf = GridSearchCV(_clf_build(clf_type), clf_params,
                                       **cv_params)


                    if self._cv_test:
                        LOG.info('Starting nested cross-validation')
                        perm_res = permutation_test_score(
                            clf, sample_x, labels_y, scoring=stype, cv=outer_cv,
                            n_permutations=self.n_perm, groups=folds_groups,
                            n_jobs=self.n_jobs)
                        LOG.info('Permutation test finished. Estimated classification score %s'
                                 ' (p-value=%s)', perm_res[0], perm_res[2])
                        thismodel['classification_score'] = perm_res[0]
                        thismodel['permutation_scores'] = perm_res[1]
                        thismodel['pvalue'] = perm_res[2]

                        score = perm_res[0]
                        pvalue = perm_res[2]

                    else:
                        LOG.info('Starting single-loop cross-validation')
                        clf.fit(sample_x, labels_y, groups=folds_groups)
                        score = clf.best_score_
                        pvalue = 0.0

                    LOG.info('Model selection finished. Score=%s, best parameters:\n\t%s',
                             score, str(clf.best_params_))

                    thismodel['best_params'] = clf.best_params_
                    thismodel['best_score'] = clf.best_score_
                    thismodel['grid_scores'] = clf.cv_results_
                    self._models[clf_type].append(thismodel)

                    if pvalue < 0.05 and best_model < score:
                        LOG.info('Best model updated (score=%f)', score)
                        best_model = score
                        self._best_model = thismodel

        LOG.info('Cross-validation finished. Best classifier is %s-%s, with a best averaged '
                 '%s score of %f and parameters %s', self._best_model['clf_type'],
                 'zs' if self._best_model['zscored'] else 'nzs', self._best_model['scoring'],
                 self._best_model['best_score'], self._best_model['best_params'])



    def get_best_cv(self, scoring=None):
        if scoring is None:
            return self._best_model
        return self._best_model[scoring]

    def outer_loop(self):
        if self.Xtest is None:
            LOG.error('Test dataset is not set')

        best_model = self.get_best_cv()

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
        for stype in self.scores:
            score, permutation_scores, pvalue = permutation_test_score(
                clf, x, y, scoring=stype, cv=ps, n_permutations=10000)
        LOG.info('Classification score %s=%s (p-value=%s)', stype, score, pvalue)


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
