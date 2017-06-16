#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
# @Author: oesteban
# @Date:   2017-06-14 12:47:30

import numpy as np
from sklearn.model_selection import LeavePGroupsOut

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
