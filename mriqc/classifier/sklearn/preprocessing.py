#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
# @Author: oesteban
# @Date:   2017-06-08 17:11:58
"""
Extensions to the sklearn's default data preprocessing filters

"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from mriqc import logging
LOG = logging.getLogger('mriqc.classifier')


class PandasAdaptor(BaseEstimator, TransformerMixin):
    """
    Wraps a data transformation to run only in specific
    columns [`source <https://stackoverflow.com/a/41461843/6820620>`_].

    Example
    -------

        >>> from sklearn.preprocessing import StandardScaler
        >>> from mriqc.classifier.sklearn.preprocessing import PandasAdaptor
        >>> tfm = PandasAdaptor(StandardScaler(),
                                  columns=['duration', 'num_operations'])
        >>> scaled = tfm.fit_transform(churn_d)

    """

    def __init__(self, columns):
        self._columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        try:
            return X[self._columns].values
        except (IndexError, KeyError):
            return X



class ColumnsScaler(BaseEstimator, TransformerMixin):
    """
    Wraps a data transformation to run only in specific
    columns [`source <https://stackoverflow.com/a/41461843/6820620>`_].

    Example
    -------

        >>> from sklearn.preprocessing import StandardScaler
        >>> from mriqc.classifier.sklearn.preprocessing import ColumnsScaler
        >>> tfm = ColumnsScaler(StandardScaler(),
                                  columns=['duration', 'num_operations'])
        >>> scaled = tfm.fit_transform(churn_d)

    """

    def __init__(self, scaler, columns=None):
        self._scaler = scaler
        self._columns = columns

    def _numeric_cols(self, X):
        columns = self._columns
        numcols = list(X.select_dtypes([np.number]).columns.ravel())

        if not columns:
            return numcols
        return [col for col in columns if col in numcols]

    def fit(self, X, y=None):
        columns = self._numeric_cols(X)
        self._scaler.fit(X[columns], y)
        return self

    def transform(self, X, y=None):
        columns = self._numeric_cols(X)

        col_order = X.columns
        scaled_x = pd.DataFrame(self._scaler.transform(
            X[columns]), columns=columns)
        unscaled_x = X.ix[:, ~X.columns.isin(columns)]
        return pd.concat([unscaled_x, scaled_x], axis=1)[col_order]


class GroupsScaler(BaseEstimator, TransformerMixin):
    """
    Wraps a data transformation to run group-wise.

    Example
    -------

        >>> from sklearn.preprocessing import StandardScaler
        >>> from mriqc.classifier.sklearn.preprocessing import GroupsScaler
        >>> tfm = GroupsScaler(StandardScaler(), groups='site')
        >>> scaled = tfm.fit_transform(churn_d)

    """

    def __init__(self, scaler, by):
        self._base_scaler = scaler
        self._by = by
        self._scalers = {}
        self._groups = None
        self._colnames = None
        self._colmask = None

    def fit(self, X, y=None):
        self._colmask = [True] * X.shape[1]
        self._colnames = X.columns.ravel().tolist()

        # Identify batches
        groups = X[[self._by]].values.ravel().tolist()
        self._colmask[X.columns.get_loc(self._by)] = False

        # Convert groups to IDs
        glist = list(set(groups))
        self._groups = np.array([glist.index(group)
                                 for group in groups])

        for gid, batch in enumerate(list(set(groups))):
            scaler = clone(self._base_scaler)
            mask = self._groups == gid
            if not np.any(mask):
                continue
            self._scalers[batch] = scaler.fit(
                X.ix[mask, self._colmask], y)

        return self

    def transform(self, X, y=None):
        if self._by in X.columns.ravel().tolist():
            groups = X[[self._by]].values.ravel().tolist()
        else:
            groups = ['Unknown'] * X.shape[0]

        glist = list(set(groups))
        groups = np.array([glist.index(group) for group in groups])
        new_x = X.copy()
        for gid, batch in enumerate(glist):
            if batch in self._scalers:
                mask = groups == gid
                if not np.any(mask):
                    continue
                scaler = self._scalers[batch]
                new_x.ix[mask, self._colmask] = scaler.transform(
                    X.ix[mask, self._colmask], y)
            else:
                colmask = self._colmask
                del colmask[self._colnames.index(self._by)]

                scaler = clone(self._base_scaler)
                new_x.ix[:, colmask] = scaler.fit_transform(
                    X.ix[:, colmask])


        return new_x


class BatchScaler(GroupsScaler, TransformerMixin):
    """
    Wraps a data transformation to run group-wise.

    Example
    -------

        >>> from sklearn.preprocessing import StandardScaler
        >>> from mriqc.classifier.sklearn.preprocessing import BatchScaler
        >>> tfm = BatchScaler(StandardScaler(), groups='site', columns=[''])
        >>> scaled = tfm.fit_transform(churn_d)

    """

    def __init__(self, scaler, by, columns=None):
        super(BatchScaler, self).__init__(scaler, by=by)
        self._columns = columns
        self._ftmask = None

    def fit(self, X, y=None):
        # Find features mask
        self._ftmask = [True] * X.shape[1]
        if self._columns:
            self._ftmask = X.columns.isin(self._columns)

        fitmsk = self._ftmask
        fitmsk[X.columns.get_loc(self._by)] = True
        super(BatchScaler, self).fit(X[X.columns[self._ftmask]], y)
        return self

    def transform(self, X, y=None):
        new_x = X.copy()

        try:
            columns = new_x.columns.ravel().tolist()
        except AttributeError:
            columns = self._columns
            print(new_x.shape[1], len(columns), sum(self._ftmask))

        if not self._by in columns:
            new_x[self._by] = ['Unknown'] * new_x.shape[0]

        new_x.ix[:, self._ftmask] = super(BatchScaler, self).transform(
            new_x[new_x.columns[self._ftmask]], y)
        return new_x


class CustFsNoiseWinnow(BaseEstimator, TransformerMixin):
    """
    Remove features with less importance than a noise feature
    https://gist.github.com/satra/c6eb113055810f19709fa7c5ebd23de8

    """
    def __init__(self):
        self.importances_ = None
        self.importances_snr_ = None
        self.idx_keep_ = None
        self.mask_ = None

    def fit(self, X, y):
        """Fit the model with X.
        This is the workhorse function.
        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
            Training data, where n_samples in the number of samples
            and n_features is the number of features.
        Returns
        -------
        self : object
            Returns the instance itself.
        self.mask_ : array
            Logical array of features to keep
        """
        from sklearn.metrics import roc_auc_score
        from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
        n_winnow = 10
        clf_flag = True
        n_estimators = 1000

        X_input = X.copy()

        n_sample, n_feature = np.shape(X_input)
        # Add "1" to the col dimension to account for always keeping the noise
        # vector inside the loop
        idx_keep = np.arange(n_feature + 1)

        counter = 0
        noise_flag = True
        while noise_flag:
            counter = counter + 1

            # Keep regenerating a noise vector as long as the noise vector is more than 0.05
            # correlated with the output.
            # Use correlation if regression, and ROC AUC if classification
            if clf_flag:
                noise_feature = np.random.normal(
                    loc=0, scale=10.0, size=(n_sample, 1))
                noise_score = roc_auc_score(
                    y, noise_feature, average='macro', sample_weight=None)
                while (noise_score > 0.6) or (noise_score < 0.4):
                    noise_feature = np.random.normal(
                        loc=0, scale=10.0, size=(n_sample, 1))
                    noise_score = roc_auc_score(
                        y, noise_feature, average='macro', sample_weight=None)
            else:
                noise_feature = np.random.normal(
                    loc=0, scale=10.0, size=(n_sample, 1))
                while np.abs(np.corrcoef(noise_feature, y[:, np.newaxis], rowvar=0)[0][1]) > 0.05:
                    noise_feature = np.random.normal(
                        loc=0, scale=10.0, size=(n_sample, 1))

            # Add noise feature
            X = np.concatenate((X_input, noise_feature), axis=1)

            # Initialize estimator
            if clf_flag:
                clf = ExtraTreesClassifier(
                    n_estimators=n_estimators,
                    criterion='gini',
                    max_depth=None,
                    min_samples_split=2, min_samples_leaf=1,
                    min_weight_fraction_leaf=0.0,
                    max_features='auto', max_leaf_nodes=None,
                    min_impurity_split=1e-07, bootstrap=False,
                    oob_score=False, n_jobs=1, random_state=None, verbose=0,
                    warm_start=False, class_weight=None)
            else:
                clf = ExtraTreesRegressor(
                    n_estimators=n_estimators, criterion='mse', max_depth=None, min_samples_split=2,
                    min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto',
                    max_leaf_nodes=None, min_impurity_split=1e-07, bootstrap=False,
                    oob_score=False, n_jobs=1, random_state=None, verbose=0, warm_start=False)

            clf.fit(X[:, idx_keep], y)
            LOG.debug('done fitting once')
            importances = clf.feature_importances_

            k = 1
            if np.all(importances[0:-1] > k*importances[-1]):
                LOG.debug('all good')
                # all features better than noise
                # comment out to force counter renditions of winnowing
                #noise_flag = False
            elif np.all(k*importances[-1] > importances[0:-1]):
                LOG.debug('all bad')
                # noise better than all features aka no feature better than noise
                # Leave as separate if clause in case want to do something different than when all feat > noise
                # comment out to force counter renditions of winnowing
                # noise_flag = False # just take everything
            else:
                LOG.debug('some good')
                # Tracer()()
                idx_keep = idx_keep[importances >= (k * importances[-1])]
                # use >= so when saving, can always drop last index
                importances = importances[importances >= (k * importances[-1])]
                # always keep the noise index, which is n_feature (assuming 0 based python index)
                #idx_keep = np.concatenate((idx_keep[:, np.newaxis], np.array([[n_feature]])), axis=0)
                idx_keep = np.ravel(idx_keep)
                LOG.debug('Feature selection: keep %d features', len(idx_keep))

            # fail safe
            if counter >= n_winnow:
                noise_flag = False

        self.importances_ = importances[:-1]
        self.importances_snr_ = importances[:-1]/importances[-1]
        self.idx_keep_ = idx_keep[:-1]
        self.mask_ = np.asarray(
            [True if i in idx_keep[:-1] else False for i in range(n_feature)])

        LOG.info('Feature selection: %d features survived', np.sum(self.mask_))
        return self

    def fit_transform(self, X, y=None):
        """Fit the model with X and apply the dimensionality reduction on X.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.
        Returns
        -------
        X_new : array-like, shape (n_samples, n_components)
        """
        self = self.fit(X, y)
        return X[:, ~self.mask_]

    def transform(self, X, y=None):
        """Apply dimensionality reduction to X.
        X is masked.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            New data, where n_samples is the number of samples
            and n_features is the number of features.
        Returns
        -------
        X_new : array-like, shape (n_samples, n_components)
        """
        from sklearn.utils import check_array
        from sklearn.utils.validation import check_is_fitted
        check_is_fitted(self, ['mask_'], all_or_any=all)
        X = check_array(X)
        return X[:, ~self.mask_]
