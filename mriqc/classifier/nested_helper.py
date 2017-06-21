#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
# @Author: oesteban
# @Date:   2017-06-15 15:42:13

from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import pandas as pd

from .sklearn.cv_nested import nested_fit_and_score, ModelAndGridSearchCV
from .sklearn._split import RobustLeavePGroupsOut as LeavePGroupsOut

from sklearn.base import is_classifier, clone
from sklearn.metrics.scorer import check_scoring
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection._split import check_cv

from .helper import CVHelperBase
from .. import logging

from builtins import str

LOG = logging.getLogger('mriqc.classifier')
LOG.setLevel(logging.INFO)

DEFAULT_TEST_PARAMETERS = {
    'svc_linear': [{'C': [0.1, 1]}],
}

EXCLUDE_COLUMNS = [
    'size_x', 'size_y', 'size_z',
    'spacing_x', 'spacing_y', 'spacing_z',
    'qi_1', 'qi_2',
    'tpm_overlap_csf', 'tpm_overlap_gm', 'tpm_overlap_wm',
]

FEATURE_NAMES = [
    'cjv', 'cnr', 'efc', 'fber',
    'fwhm_avg', 'fwhm_x', 'fwhm_y', 'fwhm_z',
    'icvs_csf', 'icvs_gm', 'icvs_wm',
    'inu_med', 'inu_range',
    'qi_1', 'qi_2',
    'rpve_csf', 'rpve_gm', 'rpve_wm',
    'size_x', 'size_y', 'size_z',
    'snr_csf', 'snr_gm', 'snr_total', 'snr_wm',
    'snrd_csf', 'snrd_gm', 'snrd_total', 'snrd_wm',
    'spacing_x', 'spacing_y', 'spacing_z',
    'summary_bg_k', 'summary_bg_mad', 'summary_bg_mean', 'summary_bg_median', 'summary_bg_n', 'summary_bg_p05', 'summary_bg_p95', 'summary_bg_stdv',
    'summary_csf_k', 'summary_csf_mad', 'summary_csf_mean', 'summary_csf_median', 'summary_csf_n', 'summary_csf_p05', 'summary_csf_p95', 'summary_csf_stdv',
    'summary_gm_k', 'summary_gm_mad', 'summary_gm_mean', 'summary_gm_median', 'summary_gm_n', 'summary_gm_p05', 'summary_gm_p95', 'summary_gm_stdv',
    'summary_wm_k', 'summary_wm_mad', 'summary_wm_mean', 'summary_wm_median', 'summary_wm_n', 'summary_wm_p05', 'summary_wm_p95', 'summary_wm_stdv',
    'tpm_overlap_csf', 'tpm_overlap_gm', 'tpm_overlap_wm',
    'wm2max'
]
FEATURE_NORM = [
    'cjv', 'cnr', 'efc', 'fber', 'fwhm_avg', 'fwhm_x', 'fwhm_y', 'fwhm_z',
    'snr_csf', 'snr_gm', 'snr_total', 'snr_wm', 'snrd_csf', 'snrd_gm', 'snrd_total', 'snrd_wm',
    'summary_csf_mad', 'summary_csf_mean', 'summary_csf_median', 'summary_csf_p05', 'summary_csf_p95', 'summary_csf_stdv', 'summary_gm_k', 'summary_gm_mad', 'summary_gm_mean', 'summary_gm_median', 'summary_gm_p05', 'summary_gm_p95', 'summary_gm_stdv', 'summary_wm_k', 'summary_wm_mad', 'summary_wm_mean', 'summary_wm_median', 'summary_wm_p05', 'summary_wm_p95', 'summary_wm_stdv'
]

FEATURE_RF_CORR = [
    'cjv', 'cnr', 'efc', 'fber', 'fwhm_avg', 'fwhm_x', 'fwhm_y', 'fwhm_z', 'icvs_csf', 'icvs_gm', 'icvs_wm',
    'qi_1', 'qi_2', 'rpve_csf', 'rpve_gm', 'rpve_wm', 'snr_csf', 'snr_gm', 'snr_total', 'snr_wm',
    'snrd_csf', 'snrd_gm', 'snrd_total', 'snrd_wm',
    'summary_bg_k', 'summary_bg_stdv',
    'summary_csf_k', 'summary_csf_mad', 'summary_csf_mean', 'summary_csf_median',
    'summary_csf_p05', 'summary_csf_p95', 'summary_csf_stdv',
    'summary_gm_k', 'summary_gm_mad', 'summary_gm_mean', 'summary_gm_median',
    'summary_gm_p05', 'summary_gm_p95', 'summary_gm_stdv',
    'summary_wm_k', 'summary_wm_mad', 'summary_wm_mean', 'summary_wm_median',
    'summary_wm_p05', 'summary_wm_p95', 'summary_wm_stdv',
    'tpm_overlap_csf', 'tpm_overlap_gm', 'tpm_overlap_wm'
]


class NestedCVHelper(CVHelperBase):

    def __init__(self, X, Y, param=None, n_jobs=-1, site_label='site', rate_label='rater_1',
                 task_id=None, scorer='roc_auc', multiclass=False,
                 verbosity=0):
        super(NestedCVHelper, self).__init__(X, Y, param=param, n_jobs=n_jobs,
                                             site_label='site', rate_label='rater_1',
                                             multiclass=False,
                                             verbosity=verbosity)
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

        outer_cv_scores = []
        outer_cv_acc = []

        # The inner CV loop is a grid search on clf_params
        LOG.info('Creating ModelAndGridSearchCV')
        inner_cv = ModelAndGridSearchCV(self.param, **gs_cv_params)

        # Some sklearn's validations
        scoring = check_scoring(inner_cv, scoring=self._scorer)
        cv_outer = check_cv(_cv_build(self.cv_outer), y,
                            classifier=is_classifier(inner_cv))

        # Outer CV loop
        LOG.info('Starting nested cross-validation ...')
        split_id = 0
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
                 np.mean(zscore_cv_auc[best_zs]), 2 *
                 np.std(zscore_cv_auc[best_zs]),
                 np.mean(zscore_cv_acc[best_zs]), 2 *
                 np.std(zscore_cv_acc[best_zs]),
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
