#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: oesteban
# @Date:   2015-11-19 16:44:27
# @Last Modified by:   oesteban
# @Last Modified time: 2017-03-07 19:39:20

"""

============================================================
:mod:`mriqc.classifier.cv` -- MRIQC Cross-validation Helpers
============================================================


"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import pandas as pd

from mriqc import __version__, logging
from .data import read_iqms, read_dataset, zscore_dataset
from .sklearn_extension import ModelAndGridSearchCV, RobustGridSearchCV, nested_fit_and_score

from sklearn.base import is_classifier, clone
from sklearn.metrics.scorer import check_scoring
from sklearn.model_selection import LeavePGroupsOut, StratifiedKFold
from sklearn.model_selection._split import check_cv

from builtins import object, str

LOG = logging.getLogger('mriqc.classifier')

DEFAULT_TEST_PARAMETERS = {
    'svc_linear': [{'C': [0.1, 1]}],
}

EXCLUDE_COLUMNS = ['size_x', 'size_y', 'size_z', 'spacing_x', 'spacing_y', 'spacing_z']

class CVHelperBase(object):

    def __init__(self, X, Y, param=None, n_jobs=-1, site_label='site', rate_label='rater_1'):
        # Initialize some values
        self.param = DEFAULT_TEST_PARAMETERS.copy()
        if param is not None:
            self.param = param

        self.n_jobs = n_jobs
        self._rate_column = rate_label
        self._site_column = site_label

        self._Xtrain, self._ftnames = read_dataset(X, Y, rate_label=rate_label)
        self.sites = list(set(self._Xtrain[site_label].values.ravel()))

    @property
    def ftnames(self):
        return self._ftnames


    @property
    def rate_column(self):
        return self._rate_column

    def fit(self):
        raise NotImplementedError

    def predict_dataset(self, data, out_file=None):
        raise NotImplementedError

    def predict(self, data):
        raise NotImplementedError

    def get_groups(self):
        groups = list(self._Xtrain[[self._site_column]].values.ravel())
        group_names = list(set(groups))
        groups_idx = []
        for g in groups:
            groups_idx.append(group_names.index(g))

        return groups_idx

    def _generate_sample(self, zscored=False):
        from sklearn.utils import indexable
        X = self._Xtr_zs.copy() if zscored else self._Xtrain.copy()
        sample_x = np.array([tuple(x) for x in X[self._ftnames].values])
        labels_y = X[[self._rate_column]].values.ravel()

        return indexable(sample_x, labels_y, self.get_groups())

class NestedCVHelper(CVHelperBase):

    def __init__(self, X, Y, param=None, n_jobs=-1, site_label='site', rate_label='rater_1',
                 task_id=None):
        super(NestedCVHelper, self).__init__(X, Y, param=param, n_jobs=n_jobs,
                                             site_label='site', rate_label='rater_1')

        self._Xtr_zs = zscore_dataset(self._Xtrain, njobs=n_jobs,
                                      excl_columns=[rate_label] + EXCLUDE_COLUMNS)
        self._models = []
        self._best_clf = {}
        self._best_model = {}
        self._cv_inner = {'type': 'kfold', 'n_splits': 10}
        self._cv_outer = None
        self._cv_scores_df = None
        self._task_id = task_id

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
    def best_clf(self):
        return self._best_clf

    @property
    def best_model(self):
        return self._best_model

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

                result = nested_fit_and_score(
                    clone(inner_cv), X, y, scoring, train, test, fit_params=fit_params, verbose=1)

                # Test group has no positive cases
                if result is None:
                    continue

                score, clf = result
                test_group = list(set(groups[i] for i in test))[0]
                self._models.append({
                    # 'clf_type': clf_str,
                    'zscored': int(dozs),
                    'outer_split_id': split_id,
                    'left-out-sites': self.sites[test_group],
                    'best_model': clf.best_model_,
                    'best_params': clf.best_params_,
                    'best_score': clf.best_score_,
                    'best_index': clf.best_index_,
                    'cv_results': clf.cv_results_,
                    'cv_scores': score['test']['roc_auc'],
                    'cv_accuracy': score['test']['accuracy'],
                    'cv_params': clf.cv_results_['params'],
                    'cv_auc_means': clf.cv_results_['mean_test_score'],
                    'cv_splits': {'split%03d' % i: clf.cv_results_['split%d_test_score' % i]
                                  for i in list(range(clf.n_splits_))}
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

    def get_inner_cv_scores(self):
        # Compose a dataframe object
        columns = ['split_id', 'zscored', 'clf', 'mean_auc', 'params']
        cvdict = {col: [] for col in columns}
        cvdict.update({key: [] for key in self._models[0]['cv_splits'].keys()})

        for model in self._models:
            for i, param in enumerate(model['cv_params']):
                cvdict['clf'] += [param[0]]
                cvdict['split_id'] += [model['outer_split_id']]
                cvdict['zscored'] += [int(model['zscored'])]
                cvdict['params'] += [param[1]]
                cvdict['mean_auc'] += [model['cv_auc_means'][i]]
                for key, val in list(model['cv_splits'].items()):
                    cvdict[key] += [val[i]]

        # massage columns
        if self._task_id is not None:
            cvdict['task_id'] = [self._task_id] * len(cvdict['clf'])
            columns.insert(0, 'task_id')

        self._cv_scores_df = pd.DataFrame(cvdict)[columns]
        return self._cv_scores_df

    def get_outer_cv_scores(self):
        # Compose a dataframe object
        columns = ['split_id', 'site', 'zscored', 'auc', 'acc']
        cvdict = {col: [] for col in columns}

        for model in self._models:
            cvdict['zscored'] += [int(model['zscored'])]
            cvdict['split_id'] += [model['outer_split_id']]
            cvdict['site'] += [model['left-out-sites']]
            cvdict['auc'] += [model['cv_scores']]
            cvdict['acc'] += [model['cv_accuracy']]

        if self._task_id is not None:
            cvdict['task_id'] = [self._task_id] * len(cvdict['split_id'])
            columns.insert(0, 'task_id')

        return pd.DataFrame(cvdict)[columns]


class CVHelper(CVHelperBase):
    def __init__(self, X=None, Y=None, load_clf=None, param=None, n_jobs=-1,
                 site_label='site', rate_label='rater_1', zscored=False):

        if (X is None or Y is None) and load_clf is None:
            raise RuntimeError('Either load_clf or X & Y should be supplied')

        self._estimator = None
        self._Xtest = None
        self._zscored = zscored
        self._pickled = False
        self._rate_column = rate_label

        if load_clf is not None:
            self.n_jobs = n_jobs
            self.load(load_clf)
            self._ftnames = getattr(self._estimator, '_ftnames')
        else:
            super(CVHelper, self).__init__(
                X, Y, param=param, n_jobs=n_jobs,
                site_label=site_label, rate_label=rate_label)
            if zscored:
                self._Xtrain = zscore_dataset(self._Xtrain, njobs=n_jobs,
                                        excl_columns=[rate_label] + EXCLUDE_COLUMNS)



    @property
    def estimator(self):
        return self._estimator


    @property
    def Xtest(self):
        return self._Xtest

    def setXtest(self, X, Y):
        self._Xtest, _ = read_dataset(X, Y, rate_label=self._rate_column)
        if self._zscored:
            self._Xtest = zscore_dataset(self._Xtest, njobs=self.n_jobs,
                                         excl_columns=[self._rate_column] + EXCLUDE_COLUMNS)

    def fit(self):
        from sklearn.ensemble import RandomForestClassifier as RFC
        if self._pickled:
            LOG.info('Classifier was loaded from file, cancelling fitting.')
            return

        LOG.info('Start fitting ...')
        estimator = RFC()
        grid = RobustGridSearchCV(
            estimator, self.param['rfc'], error_score=0.5, refit=True,
            scoring=check_scoring(estimator, scoring='roc_auc'),
            n_jobs=self.n_jobs, cv=LeavePGroupsOut(n_groups=1), verbose=0)

        X, y, groups = self._generate_sample()
        self._estimator = grid.fit(X, y, groups=groups)

        LOG.info('Model selection - best parameters (roc_auc=%f) %s',
                 grid.best_score_, grid.best_params_)

    def save(self, filehandler, compress=3):
        """
        Pickle the estimator, adding the feature names
        http://scikit-learn.org/stable/modules/model_persistence.html

        """
        from sklearn.externals.joblib import dump as savepkl

        # Store ftnames
        setattr(self._estimator, '_ftnames', self._ftnames)
        savepkl(self._estimator, filehandler, compress=compress)

    def load(self, filehandler):
        """
        UnPickle the estimator, adding the feature names
        http://scikit-learn.org/stable/modules/model_persistence.html

        """
        from sklearn.externals.joblib import load as loadpkl
        self._estimator = loadpkl(filehandler)
        self._ftnames = getattr(self._estimator, '_ftnames')
        self._pickled = True


    def predict(self, datapoints):
        return self.estimator.predict(datapoints).astype(int)

    def predict_dataset(self, data, out_file=None):
        _xeval, _, bidts = read_iqms(data)
        sample_x = np.array([tuple(x) for x in _xeval[self._ftnames].values])

        pred = _xeval[bidts].copy()
        pred['prediction'] = self.predict(sample_x).astype(int)

        if out_file is not None:
            pred[bidts + ['prediction']].to_csv(out_file, index=False)
        return pred

    def evaluate(self, scoring='accuracy'):
        from sklearn.model_selection._validation import _score

        sample_x = np.array([tuple(x) for x in self._Xtest[self._ftnames].values])
        return _score(self._estimator, sample_x, self._Xtest.rate.values.ravel().tolist(),
                      check_scoring(self._estimator, scoring=scoring))


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
