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

    def fit(self, X, y=None):
        if self._columns:
            self._scaler.fit(X[self._columns], y)
        else:
            self._scaler.fit(X, y)
        return self

    def transform(self, X, y=None):
        if not self._columns:
            return self._scaler.transform(X)

        col_order = X.columns
        scaled_x = pd.DataFrame(self._scaler.transform(
            X[self._columns]), columns=self._columns)
        unscaled_x = X.ix[:, ~X.columns.isin(self._columns)]
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

    def __init__(self, scaler, groups):
        self._base_scaler = scaler
        self._groups = groups
        self._scaler = []

    def fit(self, X, y=None):
        groups, ngroups, columns = self._get_groups(X)

        for gid in list(range(ngroups)):
            mask = groups == gid
            scaler = clone(self._base_scaler)
            scaler.fit(X.iloc[mask, columns], y)
            self._scaler.append(scaler)

        return self

    def transform(self, X, y=None):
        groups, _, columns = self._get_groups(X)

        dataframes = []
        for gid, scaler in enumerate(self._scaler):
            mask = groups == gid

            scaled_x = pd.DataFrame(scaler.transform(
                X.iloc[mask, columns]))
            dataframes.append(scaled_x)

        scaled = pd.concat(dataframes, axis=0)

        if isinstance(self._groups, str):
            scaled[self._groups] = X[[self._groups]].values

        return scaled

    def _get_groups(self, X):
        columns = X.columns.ravel().tolist()
        groups = self._groups
        if isinstance(self._groups, str):
            groups = X[[self._groups]].values.ravel().tolist()
            columns.remove(self._groups)

        glist = list(set(groups))
        ngroups = len(glist)
        groups = np.array([glist.index(group) for group in groups])
        return groups, ngroups, columns


class BatchScaler(BaseEstimator, TransformerMixin):
    """
    Wraps a data transformation to run group-wise.

    Example
    -------

        >>> from sklearn.preprocessing import StandardScaler
        >>> from mriqc.classifier.sklearn.preprocessing import BatchScaler
        >>> tfm = BatchScaler(StandardScaler(), groups='site', columns=[''])
        >>> scaled = tfm.fit_transform(churn_d)

    """

    def __init__(self, scaler, groups, columns=None):
        self._scaler = ColumnsScaler(GroupsScaler(scaler, groups=groups),
                                     columns=columns)

    def fit(self, X, y=None):
        return self._scaler.fit(X, y)

    def transform(self, X, y=None):
        return self._scaler.transform(X, y)
