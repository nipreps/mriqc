#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: oesteban
# @Date:   2015-11-19 16:44:27

"""
=====================
Extensions to sklearn
=====================


Extends sklearn's GridSearchCV to a model search object


"""
from __future__ import absolute_import, division, print_function, unicode_literals
import time

from functools import partial
from collections import Sized
import numpy as np

from sklearn.base import is_classifier, clone
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection._split import check_cv
from sklearn.model_selection._search import (
    check_scoring, indexable,
    Parallel, delayed, defaultdict, rankdata
)
from sklearn.model_selection._validation import (
    _score, _num_samples, _index_param_value, _safe_split,
    logger)

from ... import logging
from builtins import object, zip
try:
    from sklearn.utils.fixes import MaskedArray
except ImportError:
    from numpy.ma import MaskedArray

LOG = logging.getLogger('mriqc.classifier')

class RobustGridSearchCV(GridSearchCV):
    def _fit(self, X, y, groups, parameter_iterable):
        """Actual fitting,  performing the search over parameters."""
        cv = check_cv(self.cv, y, classifier=is_classifier(self.estimator))
        self.scorer_ = check_scoring(self.estimator, scoring=self.scoring)

        X, y, groups = indexable(X, y, groups)
        n_splits = cv.get_n_splits(X, y, groups)

        if self.verbose > 0 and isinstance(parameter_iterable, Sized):
            n_candidates = len(parameter_iterable)
            LOG.log(19, "Fitting %d folds for each of %d candidates, totalling"
                    " %d fits", n_splits, n_candidates, n_candidates * n_splits)
        pre_dispatch = self.pre_dispatch

        cv_iter = list(cv.split(X, y, groups))
        out = Parallel(
            n_jobs=self.n_jobs, verbose=self.verbose,
            pre_dispatch=pre_dispatch
        )(delayed(_robust_fit_and_score)(clone(self.estimator), X, y, self.scorer_,
                                  train, test, self.verbose, parameters,
                                  fit_params=self.fit_params,
                                  return_train_score=self.return_train_score,
                                  return_n_test_samples=True,
                                  return_times=True, return_parameters=True,
                                  error_score=self.error_score)
          for parameters in parameter_iterable
          for train, test in cv_iter)

        # Clean up skipped loops
        out = [i for i in out if i is not None]

        if len(out) < (n_splits * len(parameter_iterable)):
            old_splits = n_splits
            n_splits = len(out) // len(parameter_iterable)
            LOG.warning("Some splits were skipped (%d)", old_splits - n_splits)

        # if one choose to see train score, "out" will contain train score info
        if self.return_train_score:
            (train_scores, test_scores, test_sample_counts,
             fit_time, score_time, parameters) = zip(*out)
        else:
            (test_scores, test_sample_counts,
             fit_time, score_time, parameters) = zip(*out)

        candidate_params = parameters[::n_splits]
        n_candidates = len(candidate_params)

        results = dict()

        def _store(key_name, array, weights=None, splits=False, rank=False):
            """A small helper to store the scores/times to the cv_results_"""
            array = np.array(array, dtype=np.float64).reshape(n_candidates,
                                                              n_splits)
            if splits:
                for split_i in range(n_splits):
                    results["split%d_%s"
                            % (split_i, key_name)] = array[:, split_i]

            array_means = np.average(array, axis=1, weights=weights)
            results['mean_%s' % key_name] = array_means
            # Weighted std is not directly available in numpy
            array_stds = np.sqrt(np.average((array -
                                             array_means[:, np.newaxis]) ** 2,
                                            axis=1, weights=weights))
            results['std_%s' % key_name] = array_stds

            if rank:
                results["rank_%s" % key_name] = np.asarray(
                    rankdata(-array_means, method='min'), dtype=np.int32)

        # Computed the (weighted) mean and std for test scores alone
        # NOTE test_sample counts (weights) remain the same for all candidates
        test_sample_counts = np.array(test_sample_counts[:n_splits],
                                      dtype=np.int)

        _store('test_score', test_scores, splits=True, rank=True,
               weights=test_sample_counts if self.iid else None)
        if self.return_train_score:
            _store('train_score', train_scores, splits=True)
        _store('fit_time', fit_time)
        _store('score_time', score_time)

        best_index = np.flatnonzero(results["rank_test_score"] == 1)[0]
        best_parameters = candidate_params[best_index]

        # Use one MaskedArray and mask all the places where the param is not
        # applicable for that candidate. Use defaultdict as each candidate may
        # not contain all the params
        param_results = defaultdict(partial(MaskedArray,
                                            np.empty(n_candidates,),
                                            mask=True,
                                            dtype=object))
        for cand_i, params in enumerate(candidate_params):
            for name, value in params.items():
                # An all masked empty array gets created for the key
                # `"param_%s" % name` at the first occurence of `name`.
                # Setting the value at an index also unmasks that index
                param_results["param_%s" % name][cand_i] = value

        results.update(param_results)

        # Store a list of param dicts at the key 'params'
        results['params'] = candidate_params

        self.cv_results_ = results
        self.best_index_ = best_index
        self.n_splits_ = n_splits

        if self.refit:
            # fit the best estimator using the entire dataset
            # clone first to work around broken estimators
            best_estimator = clone(self.estimator).set_params(
                **best_parameters)
            if y is not None:
                best_estimator.fit(X, y, **self.fit_params)
            else:
                best_estimator.fit(X, **self.fit_params)
            self.best_estimator_ = best_estimator
        return self


def _robust_fit_and_score(estimator, X, y, scorer, train, test, verbose,
                   parameters, fit_params, return_train_score=False,
                   return_parameters=False, return_n_test_samples=False,
                   return_times=False, error_score=0.5):
    """

    """
    logtrace = '\nCV loop {:.>108}'.format

    parameters = parameters if parameters is not None else {}
    fit_params = fit_params if fit_params is not None else {}

    LOG.log(19, logtrace(' [start]'))

    # Create split
    X_train, y_train = _safe_split(estimator, X, y, train)
    X_test, y_test = _safe_split(estimator, X, y, test, train)

    if y_test is not None:
        tp = sum(y_test)
        if tp == len(y_test) or not tp:
            LOG.debug('Fold does not have any "%s" test samples.',
                     'accept' if tp == 0 else 'exclude')

            LOG.log(19, logtrace(' [skip]'))
            return None

    # Set model parameters
    estimator.set_params(**parameters)

    # Adjust length of sample weights
    fit_params = dict([(k, _index_param_value(X, v, train))
                      for k, v in fit_params.items()])


    start_time = time.time()
    try:
        if y_train is None:
            estimator.fit(X_train, **fit_params)
        else:
            estimator.fit(X_train, y_train, **fit_params)
    except Exception:
        LOG.log(19, logtrace(' [error]'))
        raise

    fit_time = time.time() - start_time
    if len(set(y_test)) == 1:
        test_score = 0.5
    else:
        test_score = _score(estimator, X_test, y_test, scorer)
    score_time = time.time() - start_time - fit_time
    if return_train_score:
        train_score = _score(estimator, X_train, y_train, scorer)

    msg = ''
    if verbose > 2:
        msg += '\n%s* Loop started %d/%d (train/test) samples' % (
            ' ' * 4, len(X_train), len(X_test))

        if y_train is not None:
            msg += '\n%s* Imbalances %.2f%%/%.2f%% (train/test positive rate)' % (
                ' ' * 4, (1 - sum(y_train) / len(X_train)) * 100,
                (1 - sum(y_test) / len(X_test)) * 100)

        msg += '\n%s* Model parameters: %s' % (' ' * 4, str(parameters))
    if verbose > 2:
        total_time = score_time + fit_time
        msg += "\n%s* total=%s" % (' ' * 4, logger.short_format_time(total_time))
    if verbose > 3:
        msg += "\n%s* score=%f" % (' ' * 4, test_score)

    if msg:
        msg += '\n'

    msg += '{:>112}'.format(logtrace(' [done]'))

    # Make sure this long message gets printed together
    LOG.log(19, msg)

    ret = [train_score, test_score] if return_train_score else [test_score]

    if return_n_test_samples:
        ret.append(_num_samples(X_test))
    if return_times:
        ret.extend([fit_time, score_time])
    if return_parameters:
        ret.append(parameters)
    return ret
