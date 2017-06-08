#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: oesteban
# @Date:   2015-11-19 16:44:27

"""

============================================================
:mod:`mriqc.classifier.cv` -- MRIQC Cross-validation Helpers
============================================================


"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import pandas as pd

from mriqc import __version__, logging
from mriqc.viz.misc import plot_batches, plot_roc_curve

from .data import read_iqms, read_dataset, zscore_dataset, balanced_leaveout, find_iqrs, norm_iqrs
from .sklearn_extension import ModelAndGridSearchCV, RobustGridSearchCV, nested_fit_and_score

from sklearn.base import is_classifier, clone
from sklearn.metrics.scorer import check_scoring, make_scorer
from sklearn.metrics import classification_report
from sklearn.model_selection import LeavePGroupsOut, StratifiedKFold
from sklearn.model_selection._split import check_cv

from builtins import object, str

LOG = logging.getLogger('mriqc.classifier')

DEFAULT_TEST_PARAMETERS = {
    'svc_linear': [{'C': [0.1, 1]}],
}

EXCLUDE_COLUMNS = [
    'size_x', 'size_y', 'size_z',
    'spacing_x', 'spacing_y', 'spacing_z',
    'qi_1', 'qi_2',
    'tpm_overlap_csf', 'tpm_overlap_gm', 'tpm_overlap_wm',
]

class CVHelperBase(object):

    def __init__(self, X, Y, param=None, n_jobs=-1, site_label='site', rate_label='rater_1',
                 scorer='roc_auc', b_leaveout=False, multiclass=False):
        # Initialize some values
        self.param = DEFAULT_TEST_PARAMETERS.copy()
        if param is not None:
            self.param = param

        self.n_jobs = n_jobs
        self._rate_column = rate_label
        self._site_column = site_label
        self._multiclass = multiclass

        self._Xtrain, self._ftnames = read_dataset(X, Y, rate_label=rate_label,
                                                   binarize=not self._multiclass)
        self.sites = list(set(self._Xtrain[site_label].values.ravel()))
        self._scorer = scorer
        self._balanced_leaveout = True
        self._Xleftout = None

        if b_leaveout:
            self._Xtrain, self._Xleftout = balanced_leaveout(self._Xtrain)


    @property
    def ftnames(self):
        return self._ftnames


    @property
    def rate_column(self):
        return self._rate_column

    def fit(self):
        raise NotImplementedError

    def predict_dataset(self, data, out_file=None, thres=0.5):
        raise NotImplementedError

    def predict(self, X, thres=0.5):
        raise NotImplementedError

    def get_groups(self):
        groups = list(self._Xtrain[[self._site_column]].values.ravel())
        group_names = list(set(groups))
        groups_idx = []
        for g in groups:
            groups_idx.append(group_names.index(g))

        return groups_idx

    def _generate_sample(self, zscored=False, full=False):
        from sklearn.utils import indexable
        X = self._Xtr_zs.copy() if zscored else self._Xtrain.copy()
        sample_x = [tuple(x) for x in X[self._ftnames].values]
        labels_y = X[[self._rate_column]].values.ravel().tolist()

        if full:
            X = self._Xtest.copy()
            LOG.warning('Requested fitting in both train and test '
                        'datasets, appending %d examples', len(X))
            sample_x += [tuple(x) for x in X[self._ftnames].values]
            labels_y += X[[self._rate_column]].values.ravel().tolist()

        groups = None
        if not full:
            groups = self.get_groups()

        return indexable(np.array(sample_x), labels_y, groups)

class NestedCVHelper(CVHelperBase):

    def __init__(self, X, Y, param=None, n_jobs=-1, site_label='site', rate_label='rater_1',
                 task_id=None, scorer='roc_auc', b_leaveout=False, multiclass=False):
        super(NestedCVHelper, self).__init__(X, Y, param=param, n_jobs=n_jobs,
                                             site_label='site', rate_label='rater_1',
                                             b_leaveout=b_leaveout, multiclass=False)

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
            scoring = check_scoring(inner_cv, scoring=self._scorer)
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
                    'cv_scores': score['test']['score'],
                    'cv_accuracy': score['test']['accuracy'],
                    'cv_params': clf.cv_results_['params'],
                    'cv_auc_means': clf.cv_results_['mean_test_score'],
                    'cv_splits': {'split%03d' % i: clf.cv_results_['split%d_test_score' % i]
                                  for i in list(range(clf.n_splits_))}
                })

                # Store the outer loop scores
                if score['test']['score'] is not None:
                    outer_cv_scores.append(score['test']['score'])
                outer_cv_acc.append(score['test']['accuracy'])
                split_id += 1

                # LOG.info(
                #     '[%s-%szs] Outer CV: roc_auc=%f, accuracy=%f, '
                #     'Inner CV: best roc_auc=%f, params=%s. ',
                #     clf.best_model_[0], 'n' if not dozs else '',
                #     score['test']['score'] if score['test']['score'] is not None else -1.0,
                #     score['test']['accuracy'],
                #     clf.best_score_, clf.best_model_[1])

            LOG.info('Outer CV loop finished, %s=%f (+/-%f), accuracy=%f (+/-%f)',
                     self._scorer,
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
        LOG.info('CV - estimated performance: %s=%f (+/-%f), accuracy=%f (+/-%f)',
                 self._scorer,
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
                 site_label='site', rate_label='rater_1', zscored=False,
                 scorer='roc_auc', b_leaveout=False, multiclass=False):

        if (X is None or Y is None) and load_clf is None:
            raise RuntimeError('Either load_clf or X & Y should be supplied')

        self._estimator = None
        self._Xtest = None
        self._zscored = zscored
        self._pickled = False
        self._rate_column = rate_label
        self._batch_effect = None

        if load_clf is not None:
            self.n_jobs = n_jobs
            self.load(load_clf)
        else:
            super(CVHelper, self).__init__(
                X, Y, param=param, n_jobs=n_jobs,
                site_label=site_label, rate_label=rate_label, scorer=scorer,
                b_leaveout=b_leaveout, multiclass=multiclass)


            self._batch_effect = find_iqrs(
                self._Xtrain, excl_columns=[rate_label] + EXCLUDE_COLUMNS)

            plot_batches(self._Xtrain[self._ftnames], 'before.png',
                         excl_columns=[rate_label] + EXCLUDE_COLUMNS)
            self._Xtrain = norm_iqrs(self._Xtrain, self._batch_effect,
                                    excl_columns=[rate_label] + EXCLUDE_COLUMNS)
            plot_batches(self._Xtrain[self._ftnames], 'after_iqrs.png',
                         excl_columns=[rate_label] + EXCLUDE_COLUMNS)
            if zscored:
                self._Xtrain = zscore_dataset(
                    self._Xtrain, njobs=n_jobs,
                    excl_columns=[rate_label] + EXCLUDE_COLUMNS)



    @property
    def estimator(self):
        return self._estimator


    @property
    def Xtest(self):
        return self._Xtest

    def setXtest(self, X, Y):
        self._Xtest, _ = read_dataset(X, Y, rate_label=self._rate_column,
                                      binarize=not self._multiclass)
        if self._batch_effect is not None:
            self._Xtest = norm_iqrs(self._Xtest, self._batch_effect,
                                      excl_columns=[self._rate_column] + EXCLUDE_COLUMNS)

        if self._zscored:
            self._Xtest = zscore_dataset(self._Xtest, njobs=self.n_jobs,
                                         excl_columns=[self._rate_column] + EXCLUDE_COLUMNS)

    def fit_full(self):
        if self._estimator is None:
            raise RuntimeError('Model should be fit first')

        LOG.info('Fitting full model ...')
        sample_x = [tuple(x) for x in self._Xtest[self._ftnames].values]
        labels_y = self._Xtest[[self._rate_column]].values.ravel().tolist()

        clf_params = self._estimator.best_estimator_.get_params()
        best_estimator = clone(self._estimator.best_estimator_).set_params(
            **clf_params)
        best_estimator.warm_start = True
        best_estimator.fit(sample_x, labels_y)
        self._estimator = best_estimator


    def fit(self):
        from sklearn.ensemble import RandomForestClassifier as RFC
        if self._pickled:
            LOG.info('Classifier was loaded from file, cancelling fitting.')
            return

        LOG.info('Start fitting (scoring=%s)...', self._scorer)
        estimator = RFC()

        thescorer = self._scorer
        if thescorer == 'brier_score_loss':
            from sklearn.metrics import brier_score_loss
            thescorer = make_scorer(brier_score_loss, greater_is_better=False,
                                    needs_proba=True, needs_threshold=False)
        grid = RobustGridSearchCV(
            estimator, self.param['rfc'], error_score=0.5, refit=True,
            scoring=check_scoring(estimator, scoring=thescorer),
            n_jobs=self.n_jobs, cv=LeavePGroupsOut(n_groups=1), verbose=0)

        X, y, groups = self._generate_sample()
        self._estimator = grid.fit(X, y, groups=groups)

        LOG.info('Model selection - best parameters (%s=%f) %s',
                 self._scorer, grid.best_score_, grid.best_params_)

        if self._Xleftout is not None:
            from sklearn.model_selection._validation import _score

            sample_x = np.array([tuple(x) for x in self._Xleftout[self._ftnames].values])
            labels_y = self._Xleftout[[self._rate_column]].values.ravel().tolist()

            LOG.info('Testing on left-out, balanced subset ...')

            score = _score(self._estimator, sample_x, labels_y,
                           check_scoring(self._estimator, scoring=self._scorer))

            LOG.info('Performance %s=%f', self._scorer, score)

            LOG.info('Classification report:\n%s',
                     classification_report(self._estimator.predict(sample_x), labels_y,
                     target_names=["accept", "exclude"]))

    def save(self, filehandler, compress=3):
        """
        Pickle the estimator, adding the feature names
        http://scikit-learn.org/stable/modules/model_persistence.html

        """
        from sklearn.externals.joblib import dump as savepkl

        # Store ftnames
        setattr(self._estimator, '_ftnames', self._ftnames)

        # Store normalization medians
        setattr(self._estimator, '_batch_effect', self._batch_effect)

        LOG.info('Saving classifier to: %s', filehandler)
        savepkl(self._estimator, filehandler, compress=compress)

    def load(self, filehandler):
        """
        UnPickle the estimator, adding the feature names
        http://scikit-learn.org/stable/modules/model_persistence.html

        """
        from sklearn.externals.joblib import load as loadpkl
        self._estimator = loadpkl(filehandler)
        self._ftnames = getattr(self._estimator, '_ftnames')
        self._batch_effect = getattr(self._estimator, '_batch_effect', None)
        self._pickled = True


    def predict(self, X, thres=0.5):
        """Predict class for X.
        The predicted class of an input sample is a vote by the trees in
        the forest, weighted by their probability estimates. That is,
        the predicted class is the one with highest mean probability
        estimate across the trees.
        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features]
            The input samples. Internally, its dtype will be converted to
            ``dtype=np.float32``. If a sparse matrix is provided, it will be
            converted into a sparse ``csr_matrix``.
        Returns
        -------
        y : array of shape = [n_samples] or [n_samples, n_outputs]
            The predicted classes.
        """
        proba = np.array(self._estimator.predict_proba(X))[:, 1]
        return (proba > thres).astype(int)

    def predict_dataset(self, data, out_file=None, thres=0.5):
        _xeval, _, bidts = read_iqms(data)
        if self._batch_effect is not None:
            _xeval = norm_iqrs(_xeval, self._batch_effect,
                                 excl_columns=[self._rate_column] + EXCLUDE_COLUMNS)

        sample_x = np.array([tuple(x) for x in _xeval[self._ftnames].values])
        pred = _xeval[bidts].copy()
        pred['proba'] = np.array(self._estimator.predict_proba(
            sample_x))[:, 1]
        pred['prediction'] = (pred['proba'].values > thres).astype(int)
        if out_file is not None:
            pred[bidts + ['prediction', 'proba']].to_csv(out_file, index=False)

        return pred

    def evaluate(self, scoring='accuracy', matrix=False, plot_roc=False):
        from sklearn.model_selection._validation import _score

        sample_x = np.array([tuple(x) for x in self._Xtest[self._ftnames].values])
        labels_y = self._Xtest.rater_1.values.ravel().tolist()
        # print(self._estimator.decision_function(sample_x))
        thescore = _score(self._estimator, sample_x, labels_y,
                          check_scoring(self._estimator, scoring=scoring))
        LOG.info('Performance %s=%f', scoring, thescore)
        if matrix:
            LOG.info('Classification report (evaluation):\n%s', classification_report(
                self._estimator.predict(sample_x), labels_y,
                target_names=["accept", "exclude"]))


        if plot_roc:
            plot_roc_curve(
                labels_y, np.array(self._estimator.predict_proba(sample_x))[:, 1],
                'roc_iqrs.png')
        return thescore


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
