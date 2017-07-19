#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: oesteban
# @Date:   2015-11-19 16:44:27

"""

.. _cvhelper:

MRIQC Cross-validation Experiments
==================================

Experiment 0: What makes a good cross-validation split in this application?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We first wanted to understand what kind of data split we should use to benchmark
the results we get using the classifier.

In a very naive approach, we use ``mriqc_clf`` to check if there are differences
between our :abbr:`LoSo (leave-one-site-out)` splits or a sandard 10-fold.
For that, we use the arguments ``--nested_cv`` and ``--nested_cv_kfold`` respectively.
Please note we are using the ``--debug`` flag to reduce the number of hyperparameters
tested in the inner cross-validation loop. Here we are cross-validating the performance
of the model selection technique: whether we will use :abbr:`LoSo (leave-one-site-out)`
or 10-fold.

Running:

  ::

      docker run -ti -v $PWD:/scratch -w /scratch \\
                 --entrypoint=/usr/local/miniconda/bin/mriqc_clf \\
                 poldracklab/mriqc:0.9.7.clf-3.0 \\
                 --train --test --log-file --nested_cv_kfold --cv kfold -v --debug

We get the following output (filtered):

  ::

      Nested CV [avg] roc_auc=0.869013861525, accuracy=0.82746147315
      Nested CV roc_auc=0.925, 0.881, 0.764, 0.904, 0.840, 0.864, 0.883, 0.857, 0.865, 0.909.
      Nested CV accuracy=0.847, 0.874, 0.757, 0.838, 0.773, 0.855, 0.809, 0.835, 0.826, 0.862.
      ...
      CV [Best model] roc_auc=0.855578059459, mean=0.856, std=0.002.
      ...
      Ratings distribution: 190/75 (71.70%/28.30%, accept/exclude)
      Testing on evaluation (left-out) dataset ...
      Performance on evaluation set (roc_auc=0.677, accuracy=0.747)
      Predictions: 253 (accept) / 12 (exclude)
      Classification report:
                   precision    recall  f1-score   support
           accept       0.74      0.99      0.85       190
          exclude       0.83      0.13      0.23        75
      avg / total       0.77      0.75      0.67       265


Please note that the outer loop evaluated an average AUC of 0.87, an average accuracy of 83%.
Then, fitting the model (using only the inner cross-validation loop on the whole dataset)
yielded an AUC=0.85 with very small variability. However, when evaluated on the held-out
dataset, the AUC dropped to 0.68 and the accuracy to 75%.

Let's repeat the experiment, but using :abbr:`LoSo (leave-one-site-out)` in the inner loop:

  ::

      docker run -ti -v $PWD:/scratch -w /scratch \\
                 --entrypoint=/usr/local/miniconda/bin/mriqc_clf \\
                 poldracklab/mriqc:0.9.7.clf-3.0 \\
                 --train --test --log-file --nested_cv_kfold --cv loso -v --debug


We get the following output (filtered):

  ::

      Nested CV [avg] roc_auc=0.858722005549, accuracy=0.819287243874
      Nested CV roc_auc=0.908, 0.874, 0.761, 0.914, 0.826, 0.842, 0.871, 0.850, 0.835, 0.906.
      Nested CV accuracy=0.838, 0.865, 0.739, 0.829, 0.782, 0.827, 0.827, 0.807, 0.817, 0.862.
      ...
      CV [Best model] roc_auc=0.744096956862, mean=0.744, std=0.112.
      ...
      Ratings distribution: 190/75 (71.70%/28.30%, accept/exclude)
      Testing on evaluation (left-out) dataset ...
      Performance on evaluation set (roc_auc=0.706, accuracy=0.770)
      Predictions: 247 (accept) / 18 (exclude)
      Classification report:
                   precision    recall  f1-score   support

           accept       0.76      0.99      0.86       190
          exclude       0.89      0.21      0.34        75

      avg / total       0.80      0.77      0.71       265


Therefore, we see that using 10-fold for the split of the outer cross-validation loop, gives us
an average AUC of 0.86 and an accuracy of 82%. Below, we see the results of fitting that model.
In a cross-validation using :abbr:`LoSo (leave-one-site-out)`, the AUC drops to 0.744. Finally
if we test the model on our left-out dataset, the final AUC is 0.71 and the accuracy 77%.

Two more evaluations, now using :abbr:`LoSo (leave-one-site-out)` in the outer loop:

  ::

      docker run -ti -v $PWD:/scratch -w /scratch \\
                 --entrypoint=/usr/local/miniconda/bin/mriqc_clf \\
                 poldracklab/mriqc:0.9.7.clf-3.0 \\
                 --train --test --log-file --nested_cv --cv kfold -v --debug


  ::

      Nested CV [avg] roc_auc=0.710537391399, accuracy=0.759618741224
      Nested CV roc_auc=0.780, 0.716, 0.829, 0.877, 0.391, 0.632, 0.679, 0.634, 0.665, \
0.472, 0.690, 0.963, 0.917, 0.528, 0.813, 0.743, 0.751.
      Nested CV accuracy=0.796, 0.421, 0.869, 0.583, 0.852, 0.625, 0.807, 0.767, 0.703, \
0.357, 0.832, 0.964, 0.911, 0.947, 0.750, 0.870, 0.860.
      ...
      CV [Best model] roc_auc=0.872377212756, mean=0.872, std=0.019.
      ...
      Ratings distribution: 190/75 (71.70%/28.30%, accept/exclude)
      Testing on evaluation (left-out) dataset ...
      Performance on evaluation set (roc_auc=0.685, accuracy=0.762)
      Predictions: 249 (accept) / 16 (exclude)
      Classification report:
                   precision    recall  f1-score   support

           accept       0.76      0.99      0.86       190
          exclude       0.88      0.19      0.31        75

      avg / total       0.79      0.76      0.70       265


And finally :abbr:`LoSo (leave-one-site-out)` in both outer and inner loops:

  ::

      docker run -ti -v $PWD:/scratch -w /scratch \\
                 --entrypoint=/usr/local/miniconda/bin/mriqc_clf \\
                 poldracklab/mriqc:0.9.7.clf-3.0 \\
                 --train --test --log-file --nested_cv --cv loso -v --debug

  ::

      Nested CV [avg] roc_auc=0.715716013846, accuracy=0.752136647911
      Nested CV roc_auc=0.963, 0.756, 0.554, 0.685, 0.673, 0.659, 0.584, 0.764, \
0.787, 0.764, 0.883, 0.843, 0.846, 0.431, 0.599, 0.910, 0.465.
      Nested CV accuracy=0.964, 0.898, 0.852, 0.789, 0.531, 0.821, 0.767, 0.947, \
0.722, 0.842, 0.528, 0.778, 0.869, 0.357, 0.766, 0.931, 0.425.
      ...
      CV [Best model] roc_auc=0.712039797411, mean=0.712, std=0.124.
      ...
      Ratings distribution: 190/75 (71.70%%/28.30%%, accept/exclude)
      Testing on evaluation (left-out) dataset ...
      Performance on evaluation set (roc_auc=0.685, accuracy=0.766)
      Predictions: 244 (accept) / 21 (exclude)
      Classification report:
                   precision    recall  f1-score   support

           accept       0.76      0.98      0.86       190
          exclude       0.81      0.23      0.35        75

      avg / total       0.78      0.77      0.71       265


Using :abbr:`LoSo (leave-one-site-out)` in the outer loop the average AUC is not that optimistic
(0.78 using K-Fold in the inner loop and 0.71 using :abbr:`LoSo (leave-one-site-out)`). Same
stands for average accuracy (76%/75% K-Fold/:abbr:`LoSo (leave-one-site-out)` in the inner loop).

When checking these results with respect to the performance on the held out dataset, the main
interpretation that arises is that the 10-Fold cross-validation is overestimating the performance.
The features have an structure correlated with the site of origin, and the 10-Fold splits do not
represent that structure well. All the folds learn something about all sites, and thus, this
cross-validated result cannot be considered a good estimation of performance on data from unseen
sites.

"""
from __future__ import absolute_import, division, print_function, unicode_literals

import os
from datetime import datetime
import numpy as np
import pandas as pd
import re
from pkg_resources import resource_filename as pkgrf

# sklearn overrides
from .sklearn import preprocessing as mcsp
from .sklearn._split import (RobustLeavePGroupsOut as LeavePGroupsOut,
                             RepeatedBalancedKFold, RepeatedPartiallyHeldOutKFold)
from .sklearn._validation import cross_val_score, permutation_test_score

# sklearn module
from sklearn import metrics as slm
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics.scorer import check_scoring
from sklearn.model_selection import (RepeatedStratifiedKFold, GridSearchCV, RandomizedSearchCV,
                                     PredefinedSplit)
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.svm import SVC, LinearSVC
from sklearn.multiclass import OneVsRestClassifier
# xgboost
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

    def __init__(self, X, Y, param_file=None, n_jobs=-1, site_label='site',
                 rate_label=None, rate_selection='random',
                 scorer='roc_auc', multiclass=False, verbosity=0, debug=False):
        # Initialize some values
        self._param_file = param_file
        self.n_jobs = n_jobs
        self._rate_column = rate_label
        self._site_column = site_label
        self._multiclass = multiclass
        self._debug = debug

        if rate_label is None:
            rate_label = ['rater_1', 'rater_2']
        self._rate_column = rate_label[0]

        self._Xtrain, self._ftnames = read_dataset(
            X, Y, rate_label=rate_label, rate_selection=rate_selection,
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
    def __init__(self, X=None, Y=None, load_clf=None, param_file=None, n_jobs=-1,
                 site_label='site', rate_label=None, scorer='roc_auc',
                 b_leaveout=False, multiclass=False, verbosity=0, split='kfold',
                 debug=False, model='rfc', basename=None, nested_cv=False,
                 nested_cv_kfold=False, permutation_test=0):

        if (X is None or Y is None) and load_clf is None:
            raise RuntimeError('Either load_clf or X & Y should be supplied')

        self._estimator = None
        self._Xtest = None
        self._pickled = False
        self._batch_effect = None
        self._split = split
        self._leaveout = b_leaveout
        self._model = model
        self._base_name = basename
        self._nestedcv = nested_cv
        self._nestedcv_kfold = nested_cv_kfold
        self._permutation_test = permutation_test

        if load_clf is not None:
            self.n_jobs = n_jobs
            self.load(load_clf)
            self._rate_column = rate_label[0]
            self._multiclass = multiclass
            self._base_name = basename[:24]
        else:
            super(CVHelper, self).__init__(
                X, Y, param_file=param_file, n_jobs=n_jobs,
                site_label=site_label, rate_label=rate_label, scorer=scorer,
                multiclass=multiclass, verbosity=verbosity, debug=debug)

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

    def _get_model(self):
        if self._model == 'xgb':
            return XGBClassifier()

        if self._model == 'svc_rbf':
            return SVC()

        if self._model == 'svc_lin':
            return LinearSVC()

        return RFC()

    def fit(self):
        """
        Fits the cross-validation helper
        """
        if self._pickled:
            LOG.info('Classifier was loaded from file, cancelling fitting.')
            return

        if self._leaveout:
            raise NotImplementedError

        LOG.info('CV [Setting up pipeline] - scorer: %s', self._scorer)

        feat_sel = self._ftnames + ['site']

        steps = [
            ('std', mcsp.BatchRobustScaler(
                by='site', columns=[ft for ft in self._ftnames if ft in FEATURE_NORM])),
            ('sel_cols', mcsp.PandasAdaptor(columns=self._ftnames + ['site'])),
            ('ft_sites', mcsp.SiteCorrelationSelector()),
            ('ft_noise', mcsp.CustFsNoiseWinnow()),
            (self._model, self._get_model())
        ]

        if self._multiclass:
            # If multiclass: binarize labels and wrap classifier
            steps.insert(3, ('bin', LabelBinarizer()))
            steps[-1] = (steps[-1][0], OneVsRestClassifier(steps[-1][1]))

        pipe = Pipeline(steps)

        # Prepare data splits for CV
        fit_args = {}
        if self._split == 'kfold':
            kf_params = {} if not self._debug else {'n_splits': 2, 'n_repeats': 1}
            splits = RepeatedStratifiedKFold(**kf_params)
        elif self._split == 'loso':
            splits = LeavePGroupsOut(n_groups=1)
        elif self._split == 'balanced-kfold':
            kf_params = {'n_splits': 10, 'n_repeats': 3}
            if self._debug:
                kf_params = {'n_splits': 3, 'n_repeats': 1}
            splits = RepeatedBalancedKFold(**kf_params)
        elif self._split == 'batch':
            # Get test label
            test_site = list(set(self._Xtest.site.values.ravel().tolist()))[0]
            # Merge test and train
            self._Xtrain = pd.concat((self._Xtrain, self._Xtest), axis=0)
            test_mask = self._Xtrain.site.values.ravel() == test_site

            kf_params = {'n_splits': 5, 'n_repeats': 1}
            if self._debug:
                kf_params = {'n_splits': 3, 'n_repeats': 1}
            kf_params['groups'] = test_mask.astype(int).tolist()
            splits = RepeatedPartiallyHeldOutKFold(**kf_params)

        train_y = self._Xtrain[[self._rate_column]].values.ravel().tolist()
        grid = RandomizedSearchCV(
            pipe, self._get_params_dist(),
            n_iter=1 if self._debug else 50,
            error_score=0.5,
            refit=True,
            scoring=check_scoring(pipe, scoring=self._scorer),
            n_jobs=self.n_jobs,
            cv=splits,
            verbose=self._verbosity)

        if self._nestedcv or self._nestedcv_kfold:
            outer_cv = LeavePGroupsOut(n_groups=1)
            if self._nestedcv_kfold:
                outer_cv = RepeatedStratifiedKFold(n_repeats=1, n_splits=10)

            n_iter = 32 if self._model in ['svc_lin', 'xgb'] else 50
            grid = RandomizedSearchCV(
                pipe, self._get_params_dist(),
                n_iter=n_iter if not self._debug else 1,
                error_score=0.5,
                refit=True,
                scoring=check_scoring(pipe, scoring=self._scorer),
                n_jobs=self.n_jobs,
                cv=splits,
                verbose=self._verbosity)

            nested_score, group_order = cross_val_score(
                grid,
                X=self._Xtrain,
                y=train_y,
                cv=outer_cv,
                scoring=['roc_auc', 'accuracy'],
            )

            nested_means = np.average(nested_score, axis=0)
            nested_std = np.std(nested_score, axis=0)
            LOG.info('Nested CV [avg] %s=%.3f (+/-%.3f), accuracy=%.3f (+/-%.3f)', self._scorer,
                     nested_means[0], nested_std[0], nested_means[1], nested_std[1])
            LOG.info('Nested CV %s=%s.', self._scorer,
                     ', '.join('%.3f' % v for v in nested_score[:, 0].tolist()))
            LOG.info('Nested CV accuracy=%s.',
                     ', '.join('%.3f' % v for v in nested_score[:, 1].tolist()))
            LOG.info('Nested CV groups=%s', group_order)
        else:
            grid = GridSearchCV(
                pipe, self._get_params(),
                error_score=0.5,
                refit=True,
                scoring=check_scoring(pipe, scoring=self._scorer),
                n_jobs=self.n_jobs,
                cv=splits,
                verbose=self._verbosity)

        grid.fit(self._Xtrain, train_y, **fit_args)
        np.savez(os.path.abspath(self._gen_fname(suffix='cvres', ext='npz')),
                 cv_results=grid.cv_results_)

        best_pos = np.argmin(grid.cv_results_['rank_test_score'])

        # Save estimator and get its parameters
        self._estimator = grid.best_estimator_
        cvparams = self._estimator.get_params()

        LOG.info('CV [Best model] %s=%s, mean=%.3f, std=%.3f.',
                 self._scorer, grid.best_score_,
                 grid.cv_results_['mean_test_score'][best_pos],
                 grid.cv_results_['std_test_score'][best_pos],
                 )
        LOG.log(18, 'CV [Best model] parameters\n%s', cvparams)

        if cvparams.get(self._model + '__oob_score', False):
            LOG.info('CV [Best model] OOB %s=%.3f', self._scorer,
                     self._estimator.named_steps[self._model].oob_score_)

        # Report preprocessing selections
        prep_msg = ' * Robust scaling (centering): %s.\n' % (
            'enabled' if cvparams['std__with_centering'] else 'disabled')
        prep_msg += ' * Robust scaling (scaling): %s.\n' % (
            'enabled' if cvparams['std__with_scaling'] else 'disabled')
        prep_msg += ' * SiteCorrelation feature selection: %s.\n' % (
            'disabled' if cvparams['ft_sites__disable'] else 'enabled')
        prep_msg += ' * Winnow feature selection: %s.\n' % (
            'disabled' if cvparams['ft_noise__disable'] else 'enabled')

        selected = np.array(feat_sel).copy()
        if not cvparams['ft_sites__disable']:
            sitesmask = self._estimator.named_steps['ft_sites'].mask_
            selected = self._Xtrain[feat_sel].columns.ravel()[sitesmask]
        if not cvparams['ft_noise__disable']:
            winnowmask = self._estimator.named_steps['ft_noise'].mask_
            selected = selected[winnowmask]

        selected = selected.tolist()
        if 'site' in selected:
            selected.remove('site')

        LOG.info('CV [Preprocessing]:\n%s * Features selected: %s.',
                 prep_msg, ', '.join(['"%s"' % f for f in selected]))

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
                'Confusion matrix:\n%s', slm.confusion_matrix(
                    test_y, pred_y))
            LOG.info(
                'Classification report:\n%s', slm.classification_report(
                    test_y, pred_y, target_names=target_names))

        if save_pred:
            self._save_pred_table(self._Xtest, prob_y, pred_y,
                                  suffix='data-test_pred')

        if save_roc:
            plot_roc_curve(self._Xtest[[self._rate_column]].values.ravel(), prob_y,
                           self._gen_fname(suffix='data-test_roc', ext='png'))

        # Run a permutation test
        if self._permutation_test:
            # Merge test and train
            concatenated_x = pd.concat((self._Xtrain, self._Xtest), axis=0)
            concatenated_y = concatenated_x[[self._rate_column]].values.ravel().tolist()
            test_fold = [-1] * len(self._Xtrain) + [0] * len(self._Xtest)

            permutation_scores = permutation_test_score(
                self._estimator, concatenated_x, concatenated_y,
                scoring='accuracy', cv=PredefinedSplit(test_fold),
                n_permutations=self._permutation_test, n_jobs=1)

            score = scores[scoring.index('accuracy')]
            pvalue = (np.sum(permutation_scores >=
                      score) + 1.0) / (self._permutation_test + 1)
            LOG.info('Permutation test (N=%d) for accuracy score %f (pvalue=%f)',
                     self._permutation_test, score, pvalue)

        return scores

    def predict(self, X, thres=0.5, return_proba=True):
        """

        Predict class for X.
        The predicted class of an input sample is a vote by the trees in
        the forest, weighted by their probability estimates. That is,
        the predicted class is the one with highest mean probability
        estimate across the trees.

        """

        if self._model == 'svc_lin':
            from sklearn.base import clone
            from sklearn.calibration import CalibratedClassifierCV
            clf = CalibratedClassifierCV(clone(self._estimator).set_param(
                **self._estimator.get_param()))
            train_y = self._Xtrain[[self._rate_column]].values.ravel().tolist()
            self._estimator = clf.fit(self._Xtrain, train_y)

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

        columns = _xeval.columns.ravel().tolist()

        if 'site' not in columns:
            _xeval['site'] = [site] * len(_xeval)
            columns.append('site')

        # Classifier is trained with rate_1 as last column
        if 'rate_1' not in columns:
            _xeval['rate_1'] = [np.nan] * len(_xeval)
            columns.append('rate_1')

        prob_y, pred_y = self.predict(_xeval[columns])
        if save_pred:
            self._save_pred_table(_xeval, prob_y, pred_y,
                                  suffix='data-%s_pred' % site)
        return pred_y

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

        # Some baseline parameters
        baseparam = {
            'std__by': ['site'],
            'std__columns': [[ft for ft in self._ftnames if ft in FEATURE_NORM]],
            'sel_cols__columns': [self._ftnames + ['site']],
        }

        # Load in classifier parameters
        clfparams = _load_parameters(
            (pkgrf('mriqc', 'data/classifier_settings.yml')
                if self._param_file is None else self._param_file)
        )

        # Read preprocessing parameters
        if 'preproc' in clfparams:
            preparams = []
            for el in clfparams['preproc']:
                pcombination = {}
                for pref, subel in list(el.items()):
                    for k, v in list(subel.items()):
                        pcombination[pref + '__' + k] = v
                preparams.append(pcombination)
        else:
            preparams = [{
                'std__with_centering': [True],
                'std__with_scaling': [True],
                'ft_sites__disable': [False],
                'ft_noise__disable': [False],
            }]

        # Set base parameters
        preparams = [{**baseparam, **prep} for prep in preparams]

        # Extract this model parameters
        prefix = self._model + '__'
        if self._multiclass:
            prefix += 'estimator__'
        modparams = {prefix + k: v for k, v in list(clfparams[self._model][0].items())}

        # Merge model parameters + preprocessing
        modparams = [{**prep, **modparams} for prep in preparams]

        # Evaluate just one model if debug
        if self._debug:
            modparams = {k: [v[0]] for k, v in list(modparams.items())}

        return modparams

    def _get_params_dist(self):
        preparams = {
            'std__by': ['site'],
            'std__with_centering': [True, False],
            'std__with_scaling': [True, False],
            'std__columns': [[ft for ft in self._ftnames if ft in FEATURE_NORM]],
            'sel_cols__columns': [self._ftnames + ['site']],
            'ft_sites__disable': [False, True],
            'ft_noise__disable': [False, True],
        }

        prefix = self._model + '__'
        if self._multiclass:
            prefix += 'estimator__'

        clfparams = _load_parameters(
            (pkgrf('mriqc', 'data/model_selection.yml')
                if self._param_file is None else self._param_file)
        )
        modparams = {prefix + k: v for k, v in list(clfparams[self._model][0].items())}
        if self._debug:
            preparams = {
                'std__by': ['site'],
                'std__with_centering': [True],
                'std__with_scaling': [True],
                'std__columns': [[ft for ft in self._ftnames if ft in FEATURE_NORM]],
                'sel_cols__columns': [self._ftnames + ['site']],
                'ft_sites__disable': [True],
                'ft_noise__disable': [True],
            }
            modparams = {k: [v[0]] for k, v in list(modparams.items())}

        return {**preparams, **modparams}


def _load_parameters(param_file):
    """Load parameters from file"""
    import yaml
    from io import open
    with open(param_file) as paramfile:
        parameters = yaml.load(paramfile)
    return parameters
