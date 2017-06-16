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

import os
from datetime import datetime
import numpy as np
import pandas as pd
import re
# sklearn overrides
from .sklearn import preprocessing as mcsp
from .sklearn._split import RobustLeavePGroupsOut as LeavePGroupsOut
# sklearn module
from sklearn import metrics as slm
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics.scorer import check_scoring
from sklearn.model_selection import RepeatedStratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.multiclass import OneVsRestClassifier
# xgboost
# from xgboost.sklearn import XGBModel as XGBClassifier
from xgboost import XGBClassifier

from .. import __version__, logging
from .data import read_dataset, get_bids_cols
from ..viz.misc import plot_roc_curve

from builtins import object

LOG = logging.getLogger('mriqc.classifier')
LOG.setLevel(logging.INFO)

FEATURE_NORM = [
    'cjv', 'cnr', 'efc', 'fber', 'fwhm_avg', 'fwhm_x', 'fwhm_y', 'fwhm_z',
    'snr_csf', 'snr_gm', 'snr_total', 'snr_wm', 'snrd_csf', 'snrd_gm', 'snrd_total', 'snrd_wm',
    'summary_csf_mad', 'summary_csf_mean', 'summary_csf_median',
    'summary_csf_p05', 'summary_csf_p95', 'summary_csf_stdv',
    'summary_gm_k', 'summary_gm_mad', 'summary_gm_mean', 'summary_gm_median',
    'summary_gm_p05', 'summary_gm_p95', 'summary_gm_stdv',
    'summary_wm_k', 'summary_wm_mad', 'summary_wm_mean', 'summary_wm_median',
    'summary_wm_p05', 'summary_wm_p95', 'summary_wm_stdv'
]


class CVHelperBase(object):
    """
    A base helper to build cross-validation schemes
    """

    def __init__(self, X, Y, param=None, n_jobs=-1, site_label='site', rate_label='rater_1',
                 scorer='roc_auc', multiclass=False, verbosity=0, debug=False):
        # Initialize some values
        self.param = param
        self.n_jobs = n_jobs
        self._rate_column = rate_label
        self._site_column = site_label
        self._multiclass = multiclass
        self._debug = debug

        self._Xtrain, self._ftnames = read_dataset(X, Y, rate_label=rate_label,
                                                   binarize=not self._multiclass)
        self.sites = list(set(self._Xtrain[site_label].values.ravel()))
        self._scorer = scorer
        self._balanced_leaveout = True
        self._verbosity = verbosity

    @property
    def ftnames(self):
        return self._ftnames

    @property
    def rate_column(self):
        return self._rate_column

    def fit(self):
        raise NotImplementedError

    def predict_dataset(self, data, thres=0.5, save_pred=False, site=None):
        raise NotImplementedError

    def predict(self, X, thres=0.5, return_proba=True):
        raise NotImplementedError


class CVHelper(CVHelperBase):
    def __init__(self, X=None, Y=None, load_clf=None, param=None, n_jobs=-1,
                 site_label='site', rate_label='rater_1', scorer='roc_auc',
                 b_leaveout=False, multiclass=False, verbosity=0, kfold=False,
                 debug=False, model='rfc'):

        if (X is None or Y is None) and load_clf is None:
            raise RuntimeError('Either load_clf or X & Y should be supplied')

        self._estimator = None
        self._Xtest = None
        self._pickled = False
        self._rate_column = rate_label
        self._batch_effect = None
        self._kfold = kfold

        if load_clf is not None:
            self.n_jobs = n_jobs
            self.load(load_clf)
        else:
            super(CVHelper, self).__init__(
                X, Y, param=param, n_jobs=n_jobs,
                site_label=site_label, rate_label=rate_label, scorer=scorer,
                multiclass=multiclass, verbosity=verbosity, debug=debug)

        self._leaveout = b_leaveout
        self._model = model
        self._base_name = 'mclf_run-%s_mod-%s_ver-%s_class-%d_cv-%s' % (
            datetime.now().strftime('%Y%m%d-%H%M%S'),
            self._model,
            re.sub('[\+_@]', '.', __version__),
            3 if self._multiclass else 2,
            'kfold' if self._kfold else 'loso',
        )

    @property
    def estimator(self):
        return self._estimator

    @property
    def Xtest(self):
        return self._Xtest

    def setXtest(self, X, Y):
        self._Xtest, _ = read_dataset(X, Y, rate_label=self._rate_column,
                                      binarize=not self._multiclass)
        if 'site' not in self._Xtest.columns.ravel().tolist():
            self._Xtest['site'] = ['TestSite'] * len(self._Xtest)

    def _gen_fname(self, suffix=None, ext=None):
        if ext is None:
            ext = ''
        if suffix is None:
            suffix = ''

        if not ext.startswith('.'):
            ext = '.' + ext

        if not suffix.startswith('_'):
            suffix = '_' + suffix

        return self._base_name + suffix + ext

    def fit(self):
        """
        Fits the cross-validation helper
        """
        if self._pickled:
            LOG.info('Classifier was loaded from file, cancelling fitting.')
            return

        train_y = self._Xtrain[[self._rate_column]].values.ravel().tolist()
        if self._multiclass:
            train_y = LabelBinarizer().fit_transform(train_y)

        if self._leaveout:
            raise NotImplementedError

        LOG.info('CV [Setting up pipeline] - Results: %s',
                 os.path.abspath(self._gen_fname(suffix='*')))

        feat_sel = self._ftnames + ['site']

        steps = [
            ('std', mcsp.BatchRobustScaler(
                by='site', columns=[ft for ft in self._ftnames if ft in FEATURE_NORM])),
            ('sel_cols', mcsp.PandasAdaptor(columns=self._ftnames + ['site'])),
            ('ft_sites', mcsp.SiteCorrelationSelector()),
            ('ft_noise', mcsp.CustFsNoiseWinnow()),
            ('rfc', RFC())
        ]

        if self._model == 'xgb':
            steps[-1] = ('xgb', XGBClassifier())

        if self._multiclass:
            steps[-1][1] = OneVsRestClassifier(steps[-1][1])

        pipe = Pipeline(steps)

        LOG.info('Cross-validation - fitting for %s ...', self._scorer)

        fit_args = {}
        if self._kfold or self._debug:
            kf_params = {} if not self._debug else {'n_splits': 2, 'n_repeats': 1}
            folds = RepeatedStratifiedKFold(**kf_params).split(
                self._Xtrain, self._Xtrain[[self._rate_column]].values.ravel().tolist())
        else:
            fit_args['groups'] = get_groups(self._Xtrain)
            folds = LeavePGroupsOut(n_groups=1).split(
                self._Xtrain, y=self._Xtrain[[self._rate_column]].values.ravel().tolist(),
                groups=fit_args['groups'])

        grid = GridSearchCV(
            pipe, self._get_params(),
            error_score=0.5,
            refit=True,
            scoring=check_scoring(pipe, scoring=self._scorer),
            n_jobs=self.n_jobs,
            cv=folds,
            verbose=self._verbosity).fit(
                self._Xtrain, train_y, **fit_args)

        np.savez(os.path.abspath(self._gen_fname(suffix='cvres', ext='npz')),
                 cv_results=grid.cv_results_)

        best_pos = np.argmin(grid.cv_results_['rank_test_score'])
        LOG.info('CV - Best %s=%s, mean=%.3f, std=%.3f.',
                 self._scorer, grid.best_score_,
                 grid.cv_results_['mean_test_score'][best_pos],
                 grid.cv_results_['std_test_score'][best_pos],
                 )
        LOG.log(18, 'CV - best model parameters\n%s',
                grid.best_params_)

        # Save estimator and leave if done
        self._estimator = grid.best_estimator_

        # Report feature selection
        selected = np.array(feat_sel).copy()
        if not self._estimator.get_params()['ft_sites__disable']:
            sitesmask = self._estimator.named_steps['ft_sites'].mask_
            selected = self._Xtrain[feat_sel].columns.ravel()[sitesmask]
            LOG.info('CV - Features after SiteCorrelationSelector: %s',
                     ', '.join(['"%s"' % f for f in selected]))
        else:
            LOG.info('CV - Feature selection based on site was disabled.')

        if not self._estimator.get_params()['ft_noise__disable']:
            winnowmask = self._estimator.named_steps['ft_noise'].mask_
            selected = selected[winnowmask]
            LOG.info('CV - Features after Winnow: %s',
                     ', '.join(['"%s"' % f for f in selected]))
        else:
            LOG.info('CV - Feature selection based on Winnow was disabled.')

        # If leaveout, test and refit
        if self._leaveout:
            self._fit_leaveout(leaveout_x, leaveout_y)

        return self

    def _fit_leaveout(self, leaveout_x, leaveout_y):

        target_names = ['accept', 'exclude']
        if self._multiclass:
            target_names = ['exclude', 'doubtful', 'accept']

        LOG.info('Testing on left-out, balanced subset ...')

        # Predict
        _, pred_y = self.predict(leaveout_x)

        LOG.info('Classification report:\n%s',
                 slm.classification_report(leaveout_y, pred_y,
                                           target_names=target_names))
        score = self._score(leaveout_x, leaveout_y)
        LOG.info('Performance on balanced left-out (%s=%f)', self._scorer, score)

        # Rewrite clf
        LOG.info('Fitting full model (train + balanced left-out) ...')
        # Features may change the robust normalization
        # self._estimator.rfc__warm_start = True
        test_yall = self._Xtrain[[self._rate_column]].values.ravel().tolist()
        if self._multiclass:
            test_yall = LabelBinarizer().fit_transform(test_yall)
        self._estimator = self._estimator.fit(self._Xtrain, test_yall)

        LOG.info('Testing on left-out with full model, balanced subset ...')
        _, pred_y = self.predict(leaveout_x)
        LOG.info('Classification report:\n%s',
                 slm.classification_report(leaveout_y, pred_y,
                                           target_names=target_names))
        score = self._score(leaveout_x, leaveout_y)
        LOG.info('Performance on balanced left-out (%s=%f)', self._scorer, score)

    def fit_full(self):
        """
        Completes the training of the model with the examples
        from the left-out dataset
        """
        if self._estimator is None:
            raise RuntimeError('Model should be fit first')
        target_names = ["accept", "exclude"]

        X = pd.concat([self._Xtrain, self._Xtest], axis=0)
        labels_y = X[[self._rate_column]].values.ravel().tolist()

        if self._multiclass:
            labels_y = LabelBinarizer().fit_transform(labels_y)
            target_names = ["exclude", "doubtful", "accept"]

        LOG.info('Fitting full model ...')
        self._estimator = self._estimator.fit(X, labels_y)

        LOG.info('Testing on left-out with full model')
        pred_y = self._estimator.predict(X)
        LOG.info('Classification report:\n%s',
                 slm.classification_report(labels_y, pred_y,
                                           target_names=target_names))
        score = self._score(X, labels_y)
        LOG.info('Full model performance on left-out (%s=%f)', self._scorer, score)

    def evaluate(self, scoring=None, matrix=False, save_roc=False,
                 save_pred=False):
        """
        Evaluate the internal estimator on the test data
        """

        if scoring is None:
            scoring = ['accuracy']

        LOG.info('Testing on evaluation (left-out) dataset ...')
        test_y = self._Xtest[[self._rate_column]].values.ravel()

        target_names = ["accept", "exclude"]
        if self._multiclass:
            target_names = ["exclude", "doubtful", "accept"]
            test_y = LabelBinarizer().fit_transform(test_y)

        prob_y, pred_y = self.predict(self._Xtest)
        scores = [self._score(self._Xtest, test_y, scoring=s) for s in scoring]

        LOG.info('Performance on evaluation set (%s)',
                 ', '.join(['%s=%.3f' % (n, s) for n, s in zip(scoring, scores)]))

        pred_totals = np.sum(pred_y, 0).tolist()
        if prob_y.shape[1] <= 2:
            pred_totals = [len(pred_y) - pred_totals, pred_totals]

        LOG.info('Predictions: %s', ' / '.join((
            '%d (%s)' % (n, c) for n, c in zip(pred_totals, target_names))))

        if matrix:
            LOG.info(
                'Classification report:\n%s', slm.classification_report(
                    test_y, pred_y, target_names=target_names))

        if save_pred:
            self._save_pred_table(self._Xtest, prob_y, pred_y,
                                  suffix='data-test_pred')

        if save_roc:
            plot_roc_curve(self._Xtest[[self._rate_column]].values.ravel(), prob_y,
                           self._gen_fname(suffix='data-test_roc', ext='png'))
        return scores

    def predict(self, X, thres=0.5, return_proba=True):
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
        proba = np.array(self._estimator.predict_proba(X))

        if proba.shape[1] > 2:
            pred = (proba > thres).astype(int)
        else:
            pred = (proba[:, 1] > thres).astype(int)

        if return_proba:
            return proba, pred

        return pred

    def predict_dataset(self, data, thres=0.5, save_pred=False, site=None):
        from .data import read_iqms
        _xeval, _, _ = read_iqms(data)

        if site is None:
            site = 'unseen'

        if 'site' not in _xeval.columns.ravel():
            _xeval['site'] = [site] * len(_xeval)

        prob_y, pred_y = self.predict(_xeval)
        if save_pred:
            self._save_pred_table(_xeval, prob_y, pred_y,
                                  suffix='data-%s_pred' % site)

        return pred

    def _save_pred_table(self, sample, prob_y, pred_y, suffix):
        bidts = get_bids_cols(sample)
        predf = sample[bidts].copy()

        if self._multiclass:
            probs = ['proba_%d' % i
                     for i in list(range(prob_y.shape[1]))]
            predf['pred_y'] = (np.argmax(pred_y, axis=1) - 1).astype(int)

            for i, col in enumerate(probs):
                predf[col] = prob_y[:, i]

            cols = probs + ['pred_y']
        else:
            cols = ['prob_y', 'pred_y']
            predf['prob_y'] = prob_y[:, 1]
            predf['pred_y'] = pred_y

        predf[bidts + cols].to_csv(
            self._gen_fname(suffix=suffix, ext='csv'),
            index=False)

    def save(self, suffix='estimator', compress=3):
        """
        Pickle the estimator, adding the feature names
        http://scikit-learn.org/stable/modules/model_persistence.html

        """
        from sklearn.externals.joblib import dump as savepkl

        # Store ftnames
        setattr(self._estimator, '_ftnames', self._ftnames)

        # Store normalization medians
        setattr(self._estimator, '_batch_effect', self._batch_effect)

        filehandler = os.path.abspath(
            self._gen_fname(suffix=suffix, ext='pklz'))

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

    def _score(self, X, y, scoring=None, clf=None):
        from sklearn.model_selection._validation import _score

        if scoring is None:
            scoring = self._scorer

        if clf is None:
            clf = self._estimator

        return _score(clf, X, y, check_scoring(clf, scoring=scoring))

    def _get_params(self):
        preparams = [
            {
                'std__by': ['site'],
                'std__with_centering': [True],
                'std__with_scaling': [True],
                'std__columns': [[ft for ft in self._ftnames if ft in FEATURE_NORM]],
                'sel_cols__columns': [self._ftnames + ['site']],
                'ft_sites__disable': [False, True],
                'ft_noise__disable': [False, True],
            },
            {
                'std__by': ['site'],
                'std__with_centering': [True, False],
                'std__with_scaling': [True, False],
                'std__columns': [[ft for ft in self._ftnames if ft in FEATURE_NORM]],
                'sel_cols__columns': [self._ftnames + ['site']],
                'ft_sites__disable': [True],
                'ft_noise__disable': [True],
            },
        ]

        if self._debug:
            preparams = [
                {
                    'std__by': ['site'],
                    'std__with_centering': [False],
                    'std__with_scaling': [False],
                    'std__columns': [[ft for ft in self._ftnames if ft in FEATURE_NORM]],
                    'sel_cols__columns': [self._ftnames + ['site']],
                    'ft_sites__disable': [True],
                    'ft_noise__disable': [True],
                },
            ]

        prefix = self._model + '__'
        if self._multiclass:
            prefix += 'estimator__'

        modparams = {prefix + k: v for k, v in list(self.param[self._model][0].items())}
        if self._debug:
            modparams = {k: [v[0]] for k, v in list(modparams.items())}

        return [{**prep, **modparams} for prep in preparams]

def get_groups(X, label='site'):
    """Generate the index of sites"""
    groups = X[label].values.ravel().tolist()
    gnames = list(set(groups))
    return [gnames.index(g) for g in groups]
