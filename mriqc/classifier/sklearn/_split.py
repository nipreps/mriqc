#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
# @Author: oesteban
# @Date:   2017-06-14 12:47:30

import numpy as np
from sklearn.model_selection import (LeavePGroupsOut, StratifiedKFold, KFold, )
                                     # _RepeatedSplits)
from sklearn.utils import check_random_state

from ... import logging
LOG = logging.getLogger('mriqc.classifier')


class RobustLeavePGroupsOut(LeavePGroupsOut):
    """
    A LeavePGroupsOut split ensuring all folds have positive and
    negative samples.

    """

    def __init__(self, n_groups):
        self._splits = None
        super(RobustLeavePGroupsOut, self).__init__(n_groups)

    def split(self, X, y=None, groups=None):
        if self._splits:
            return self._splits

        self._splits = list(super(RobustLeavePGroupsOut, self).split(
            X, y=y, groups=groups))

        rmfold = []
        for i, (_, test_idx) in enumerate(self._splits):
            if len(np.unique(np.array(y)[test_idx])) == 1:
                rmfold.append(i)

        if rmfold:
            self._splits = [split for i, split in enumerate(self._splits)
                            if i not in rmfold]
            LOG.warning('Some splits (%d) were dropped because one or more classes'
                        ' are totally missing', len(rmfold))

        return self._splits

    def get_n_splits(self, X, y, groups):
        return len(self._splits)


class BalancedKFold(StratifiedKFold):
    """
    A balanced K-Fold split
    """

    def split(self, X, y, groups=None):
        splits = super(BalancedKFold, self).split(X, y, groups)

        for train_index, test_index in splits:
            split_y = y[test_index]
            classes_y, y_inversed = np.unique(split_y, return_inverse=True)
            min_y = min(np.bincount(y_inversed))
            new_index = np.zeros(min_y * len(classes_y), dtype=int)

            for cls in classes_y:
                cls_index = test_index[split_y == cls]
                if len(cls_index) > min_y:
                    cls_index = np.random.choice(
                        cls_index, size=min_y, replace=False)

                new_index[cls * min_y:(cls + 1) * min_y] = cls_index
            yield train_index, new_index



# class RepeatedBalancedKFold(_RepeatedSplits):
#     """
#     A repeated K-Fold split, where test folds are balanced
#     """

#     def __init__(self, n_splits=5, n_repeats=10, random_state=None):
#         super(RepeatedBalancedKFold, self).__init__(
#             BalancedKFold, n_repeats, random_state, n_splits=n_splits)
