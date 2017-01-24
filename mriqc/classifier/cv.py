#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: oesteban
# @Date:   2015-11-19 16:44:27
# @Last Modified by:   oesteban
# @Last Modified time: 2017-01-23 17:29:30

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
from .model_selection import ModelAndGridSearchCV


from sklearn.base import is_classifier, clone
from sklearn.utils import indexable
from sklearn.metrics.scorer import check_scoring
try:
    from sklearn.model_selection import (LeavePGroupsOut, StratifiedKFold, GridSearchCV,
                                         permutation_test_score, PredefinedSplit, cross_val_score)
    from sklearn.model_selection._split import check_cv
except ImportError:
    from sklearn.cross_validation import (
        permutation_test_score, StratifiedKFold,
        LeaveOneLabelOut as LeaveOneGroupOut,
        PredefinedSplit, cross_val_score)
    from sklearn.grid_search import GridSearchCV


LOG = logging.getLogger('mriqc.classifier')

DEFAULT_TEST_PARAMETERS = {
    'svc_linear': [{'C': [0.1, 1]}],
}

class CVHelper(object):

    def __init__(self, X, Y, scores=None, param=None, n_jobs=-1, n_perm=5000,
                 site_label='site', rate_label='rate'):

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
            self.X, njobs=n_jobs, excl_columns=[
                rate_label, 'size_x', 'size_y', 'size_z',
                'spacing_x', 'spacing_y', 'spacing_z'])

        self._models = []
        self._best_clf = {}
        self._best_model = {}
        self.n_perm = n_perm
        self._cv_inner = {'type': 'kfold', 'n_splits': 10}
        self._cv_outer = None
        self._cv_scores_df = None

    @property
    def cv_scores_df(self):
        return self._cv_scores_df

    @property
    def cv_inner(self):
        return self._cv_inner

    @cv_inner.setter
    def cv_inner(self, value):
        self._cv_inner = value

    @property
    def cv_outer(self):
        return self._cv_outer

    @cv_outer.setter
    def cv_outer(self, value):
        self._cv_outer = value

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

    def to_csv(self, out_file, output_set='training'):
        if output_set == 'training':
            self.X.to_csv(out_file, index=False)
        elif output_set == 'evaluation':
            self.Xtest.to_csv(out_file, index=False)

    def fit(self):
        LOG.info('Start fitting ...')

        gs_cv_params = {'n_jobs': self.n_jobs, 'cv': _cv_build(self.cv_inner),
                        'verbose': 0}

        zscore_cv_auc = []
        zscore_cv_acc = []
        split_id = 0
        for dozs in [False, True]:
            LOG.info('Generate %sz-scored sample ...', '' if dozs else 'non ')
            X, y, groups = self._generate_sample(zscored=dozs)
            # clf_str = '%s-%szs' % (clf_type.upper(), '' if dozs else 'n')
            # LOG.info('CV loop [scorer=roc_auc, classifier=%s]', clf_str)

            # The inner CV loop is a grid search on clf_params
            LOG.info('Creating ModelAndGridSearchCV')
            inner_cv = ModelAndGridSearchCV(self.param, **gs_cv_params)

            # Some sklearn's validations
            scoring = check_scoring(inner_cv, scoring='roc_auc')
            cv_outer = check_cv(_cv_build(self.cv_outer), y,
                                classifier=is_classifier(inner_cv))

            # Outer CV loop
            outer_cv_scores = []
            outer_cv_acc = []
            LOG.info('Starting nested cross-validation ...')
            for train, test in list(cv_outer.split(X, y, groups)):
                # Find the groups in the train set, in case inner CV is LOSO.
                fit_params = None
                if self.cv_inner.get('type') == 'loso':
                    train_groups = [groups[i] for i in train]
                    fit_params = {'groups': train_groups}

                result = _fit_and_score(clone(inner_cv), X, y, scoring, train, test,
                                        fit_params=fit_params, verbose=1)

                # Test group has no positive cases
                if result is None:
                    continue

                score, clf = result
                test_groups = list(set(groups[i] for i in test))
                self._models.append({
                    # 'clf_type': clf_str,
                    'zscored': int(dozs),
                    'outer_split_id': split_id,
                    'left-out-sites': [self.sites[i] for i in test_groups],
                    'best_model': clf.best_model_,
                    'best_params': clf.best_params_,
                    'best_score': clf.best_score_,
                    'best_index': clf.best_index_,
                    'cv_results': clf.cv_results_,
                    'cv_scores': score['test']['roc_auc'],
                    'cv_accuracy': score['test']['accuracy'],
                    'cv_params': clf.cv_results_['params'],
                    'cv_auc_means': clf.cv_results_['mean_test_score'],
                    'cv_splits': [clf.cv_results_['split%d_test_score' % i]
                                  for i in list(range(clf.n_splits_))]
                })

                # Store the outer loop scores
                if score['test']['roc_auc'] is not None:
                    outer_cv_scores.append(score['test']['roc_auc'])
                outer_cv_acc.append(score['test']['accuracy'])
                split_id += 1

                # LOG.info(
                #     '[%s-%szs] Outer CV: roc_auc=%f, accuracy=%f, '
                #     'Inner CV: best roc_auc=%f, params=%s. ',
                #     clf.best_model_[0], 'n' if not dozs else '',
                #     score['test']['roc_auc'] if score['test']['roc_auc'] is not None else -1.0,
                #     score['test']['accuracy'],
                #     clf.best_score_, clf.best_model_[1])

            LOG.info('Outer CV loop finished, roc_auc=%f (+/-%f), accuracy=%f (+/-%f)',
                     np.mean(outer_cv_scores), 2 * np.std(outer_cv_scores),
                     np.mean(outer_cv_acc), 2 * np.std(outer_cv_acc))

            zscore_cv_auc.append(outer_cv_scores)
            zscore_cv_acc.append(outer_cv_acc)


        # Select best performing model
        best_inner_loops = [model['best_score'] for model in self._models]
        best_idx = np.argmax(best_inner_loops)
        self._best_model = self._models[best_idx]
        LOG.info('Inner CV [%d models compared] - best model %s-%szs, score=%f, params=%s',
                 len(best_inner_loops) * len(self._models[0]['cv_params']),
                 self._best_model['best_model'][0],
                 'n' if not self._best_model['zscored'] else '',
                 self._best_model['best_score'], self._best_model['best_params'])

        # Write out evaluation result
        best_zs = 1 if self._best_model['zscored'] else 0
        LOG.info('CV - estimated performance: roc_auc=%f (+/-%f), accuracy=%f (+/-%f)',
                 np.mean(zscore_cv_auc[best_zs]), 2 * np.std(zscore_cv_auc[best_zs]),
                 np.mean(zscore_cv_acc[best_zs]), 2 * np.std(zscore_cv_acc[best_zs]),
        )

        # Compose a dataframe object
        cvdict = {
            'clf': [],
            'zscored': [],
            'params': [],
            'roc_auc': [],
            'mean_auc': [],
            'split_id': []
        }
        for model in self._models:
            for i, param in enumerate(model['cv_params']):
                loop_scores = np.array(model['cv_splits'])
                nscores = loop_scores.shape[0] # Shape should be n_splits x n_param_comb
                cvdict['clf'] += [param[0]] * nscores
                cvdict['split_id'] += [model['outer_split_id']] * nscores
                cvdict['zscored'] += [int(model['zscored'])] * nscores
                cvdict['params'] += [param[1]] * nscores
                cvdict['mean_auc'] += [model['cv_auc_means'][i]] * nscores
                cvdict['roc_auc'] += loop_scores[:, i].ravel().tolist()

        self._cv_scores_df = pd.DataFrame(cvdict)[[
            'clf', 'split_id', 'zscored', 'roc_auc', 'mean_auc', 'params']]



    def get_groups(self):
        groups = list(self.X[[self._site_column]].values.ravel())
        group_names = list(set(groups))
        groups_idx = []
        for g in groups:
            groups_idx.append(group_names.index(g))

        return groups_idx

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


    def _generate_sample(self, zscored=False):
        X = self.Xzscored.copy() if zscored else self.X.copy()
        sample_x = np.array([tuple(x) for x in X[self.ftnames].values])
        labels_y = X[[self._rate_column]].values.ravel()

        return indexable(sample_x, labels_y, self.get_groups())


def _fit_and_score(estimator, X, y, scorer, train, test, verbose=1,
                   parameters=None, fit_params=None, return_train_score=False,
                   return_times=False, error_score='raise'):
    """
    Fit estimator and compute scores for a given dataset split.

    Parameters
    ----------
    estimator : estimator object implementing 'fit'
        The object to use to fit the data.
    X : array-like of shape at least 2D
        The data to fit.
    y : array-like, optional, default: None
        The target variable to try to predict in the case of
        supervised learning.
    scorer : callable
        A scorer callable object / function with signature
        ``scorer(estimator, X, y)``.
    train : array-like, shape (n_train_samples,)
        Indices of training samples.
    test : array-like, shape (n_test_samples,)
        Indices of test samples.
    verbose : integer
        The verbosity level.
    error_score : 'raise' (default) or numeric
        Value to assign to the score if an error occurs in estimator fitting.
        If set to 'raise', the error is raised. If a numeric value is given,
        FitFailedWarning is raised. This parameter does not affect the refit
        step, which will always raise the error.
    parameters : dict or None
        Parameters to be set on the estimator.
    fit_params : dict or None
        Parameters that will be passed to ``estimator.fit``.
    return_train_score : boolean, optional, default: False
        Compute and return score on training set.
    return_parameters : boolean, optional, default: False
        Return parameters that has been used for the estimator.

    Returns
    -------
    train_score : float, optional
        Score on training set, returned only if `return_train_score` is `True`.
    test_score : float
        Score on test set.
    n_test_samples : int
        Number of test samples.
    fit_time : float
        Time spent for fitting in seconds.
    score_time : float
        Time spent for scoring in seconds.
    parameters : dict or None, optional
        The parameters that have been evaluated.
    """
    import time
    import numbers
    from sklearn.utils.metaestimators import _safe_split
    from sklearn.model_selection._validation import _index_param_value, _score
    from sklearn.externals.joblib.logger import short_format_time

    # Adjust length of sample weights
    fit_params = fit_params if fit_params is not None else {}
    fit_params = dict([(k, _index_param_value(X, v, train))
                      for k, v in fit_params.items()])

    if parameters is not None:
        estimator.set_params(**parameters)

    start_time = time.time()

    X_train, y_train = _safe_split(estimator, X, y, train)
    X_test, y_test = _safe_split(estimator, X, y, test, train)

    if verbose > 1:
        LOG.info('CV iteration: Xtrain=%d, Ytrain=%d/%d -- Xtest=%d, Ytest=%d/%d.',
                 len(X_train), len(X_train) - sum(y_train), sum(y_train),
                 len(X_test), len(X_test) - sum(y_test), sum(y_test))

    try:
        if y_train is None:
            estimator.fit(X_train, **fit_params)
        else:
            estimator.fit(X_train, y_train, **fit_params)

    except Exception as e:
        # Note fit time as time until error
        fit_time = time.time() - start_time
        score_time = 0.0
        if error_score == 'raise':
            raise
        elif isinstance(error_score, numbers.Number):
            test_score = error_score
            if return_train_score:
                train_score = error_score
            LOG.warn("Classifier fit failed. The score on this train-test"
                     " partition for these parameters will be set to %f. "
                     "Details: \n%r", error_score, e)
        else:
            raise ValueError("error_score must be the string 'raise' or a"
                             " numeric value. (Hint: if using 'raise', please"
                             " make sure that it has been spelled correctly.)")

    else:
        fit_time = time.time() - start_time

        test_score = None
        score_time = 0.0
        if len(set(y_test)) > 1:
            test_score = _score(estimator, X_test, y_test, scorer)
            score_time = time.time() - start_time - fit_time
        else:
            LOG.warn('Test set has no positive labels, scoring has been skipped '
                     'in this loop.')

        if return_train_score:
            train_score = _score(estimator, X_train, y_train, scorer)

        acc_score = _score(estimator, X_test, y_test,
                           check_scoring(estimator, scoring='accuracy'))
        inner_mean_scores = [
            estimator.cv_results_['split%d_test_score' % i][estimator.best_index_]
            for i in list(range(estimator.n_splits_))
        ]

    if verbose > 0:
        total_time = score_time + fit_time
        if test_score is not None:
            LOG.info('Iteration took %s, score=%f, accuracy=%f.',
                     short_format_time(total_time), test_score, acc_score)
        else:
            LOG.info('Iteration took %s, score=None, accuracy=%f.',
                     short_format_time(total_time), acc_score)

    ret = {
        'test': {'roc_auc': test_score, 'accuracy': acc_score,
                 'loop_scores': inner_mean_scores}
    }

    if return_train_score:
        ret['train'] = {'roc_auc': train_score}

    if return_times:
        ret['times'] = [fit_time, score_time]

    return ret, estimator

def _clf_build(clf_type):
    if clf_type == 'svc_linear':
        return svm.LinearSVC(C=1)
    elif clf_type == 'svc_rbf':
        return svm.SVC(C=1)
    elif clf_type == 'rfc':
        return RFC()

def _cv_build(cv_scheme):
    LOG.debug('Building CV scheme: %s', str(cv_scheme))
    if cv_scheme is None:
        return None

    if cv_scheme is not None and cv_scheme.get('type', '') == 'kfold':
        nsplits = cv_scheme.get('n_splits', 6)
        return StratifiedKFold(n_splits=nsplits, shuffle=True)

    if cv_scheme is not None and cv_scheme.get('type', '') == 'loso':
        return LeavePGroupsOut(n_groups=1)

    raise RuntimeError('Unknown CV scheme (%s)' % str(cv_scheme))
