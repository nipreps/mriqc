import numpy as np
from sklearn.utils import indexable
from sklearn.model_selection import LeavePGroupsOut, StratifiedKFold
from sklearn.model_selection._split import _RepeatedSplits

import logging

LOG = logging.getLogger("mriqc.classifier")


class RobustLeavePGroupsOut(LeavePGroupsOut):
    """
    A LeavePGroupsOut split ensuring all folds have positive and
    negative samples.

    """

    def __init__(self, n_groups, groups=None):
        self._splits = None
        self._groups = groups
        super(RobustLeavePGroupsOut, self).__init__(n_groups)

    def split(self, X, y=None, groups=None):
        if self._splits:
            return self._splits

        if groups is None:
            groups = self._groups

        if groups is None:
            from ..data import get_groups

            groups, _ = get_groups(X)
            self._groups = groups

        self._splits = list(
            super(RobustLeavePGroupsOut, self).split(X, y=y, groups=groups)
        )

        rmfold = []
        for i, (_, test_idx) in enumerate(self._splits):
            if len(np.unique(np.array(y)[test_idx])) == 1:
                rmfold.append(i)

        if rmfold:
            self._splits = [
                split for i, split in enumerate(self._splits) if i not in rmfold
            ]
            LOG.warning(
                "Some splits (%d) were dropped because one or more classes"
                " are totally missing",
                len(rmfold),
            )

        return self._splits

    @property
    def groups(self):
        return self._groups

    def get_n_splits(self, X, y, groups):
        return len(self.split(X, y, groups))


class BalancedKFold(StratifiedKFold):
    """
    A balanced K-Fold split
    """

    def split(self, X, y, groups=None):
        splits = super(BalancedKFold, self).split(X, y, groups)

        y = np.array(y)
        for train_index, test_index in splits:
            split_y = y[test_index]
            classes_y, y_inversed = np.unique(split_y, return_inverse=True)
            min_y = min(np.bincount(y_inversed))
            new_index = np.zeros(min_y * len(classes_y), dtype=int)

            for cls in classes_y:
                cls_index = test_index[split_y == cls]
                if len(cls_index) > min_y:
                    cls_index = np.random.choice(cls_index, size=min_y, replace=False)

                new_index[cls * min_y:(cls + 1) * min_y] = cls_index
            yield train_index, new_index


class RepeatedBalancedKFold(_RepeatedSplits):
    """
    A repeated K-Fold split, where test folds are balanced
    """

    def __init__(self, n_splits=5, n_repeats=10, random_state=None):
        super(RepeatedBalancedKFold, self).__init__(
            BalancedKFold, n_repeats, random_state, n_splits=n_splits
        )


class PartiallyHeldOutKFold(StratifiedKFold):
    """
    A K-Fold split on the test set where the train splits are
    augmented with the original train set (in whole).
    """

    def __init__(self, n_splits=3, shuffle=False, random_state=None, groups=None):
        self._splits = None
        self._groups = groups
        super(PartiallyHeldOutKFold, self).__init__(
            n_splits=n_splits, shuffle=shuffle, random_state=random_state
        )

    def split(self, X, y, groups=None):
        if groups is None:
            groups = self._groups

        X, y, groups = indexable(X, y, groups)

        msk = np.array(groups, dtype=bool)
        train_idx = np.arange(len(X))[~msk]
        test_idx = np.arange(len(X))[msk]

        try:
            test_x = X.as_matrix()[test_idx, :]
        except AttributeError:
            test_x = X[test_idx, :]

        test_y = np.array(y)[test_idx]
        split = super(PartiallyHeldOutKFold, self).split(test_x, test_y)

        offset = test_idx[0]
        for test_train, test_test in split:
            test_train = np.concatenate((train_idx, test_train + offset))
            yield test_train, test_test


class RepeatedPartiallyHeldOutKFold(_RepeatedSplits):
    """
    A repeated RepeatedPartiallyHeldOutKFold split
    """

    def __init__(self, n_splits=5, n_repeats=10, random_state=None, groups=None):
        super(RepeatedPartiallyHeldOutKFold, self).__init__(
            PartiallyHeldOutKFold,
            n_repeats,
            random_state,
            n_splits=n_splits,
            groups=groups,
        )
