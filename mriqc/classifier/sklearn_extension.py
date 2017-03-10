#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: oesteban
# @Date:   2015-11-19 16:44:27
# @Last Modified by:   oesteban
# @Last Modified time: 2017-03-09 17:17:08

"""
=====================
Extensions to sklearn
=====================


Extends sklearn's GridSearchCV to a model search object


"""
from __future__ import absolute_import, division, print_function, unicode_literals
import warnings
import numbers
import time

from functools import partial, reduce
from itertools import product
from collections import Mapping, Sized
import operator
import numpy as np

from sklearn.base import is_classifier, clone
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection._split import check_cv
from sklearn.model_selection._search import (
    BaseSearchCV, check_scoring, indexable,
    Parallel, delayed, defaultdict, rankdata
)

from sklearn.utils.fixes import np_version

from sklearn.model_selection._validation import (
    _score, _num_samples, _index_param_value, _safe_split,
    FitFailedWarning, logger)

from mriqc import logging
from builtins import object, zip

if np_version < (1, 12, 0):
    from sklearn.utils.fixes import MaskedArray
else:
    from numpy.ma import MaskedArray

LOG = logging.getLogger('mriqc.classifier')

def _len(indict):
    product = partial(reduce, operator.mul)
    return sum(product(len(v) for v in p.values()) if p else 1
               for p in indict)

class ModelParameterGrid(object):
    """
    Grid of models and parameters with a discrete number of values for each.
    Can be used to iterate over parameter value combinations with the
    Python built-in function iter.
    Read more in the :ref:`User Guide <search>`.
    Parameters
    ----------
    param_grid : dict of string to sequence, or sequence of such
        The parameter grid to explore, as a dictionary mapping estimator
        parameters to sequences of allowed values.
        An empty dict signifies default parameters.
        A sequence of dicts signifies a sequence of grids to search, and is
        useful to avoid exploring parameter combinations that make no sense
        or have no effect. See the examples below.

    Examples
    --------
    >>> from mriqc.classifier.model_selection import ModelParameterGrid
    >>> param_grid = {'model1': [{'a': [1, 2], 'b': [True, False]}], 'model2': [{'a': [0]}]}
    >>> len(ModelParameterGrid(param_grid)) == 5
    True
    >>> list(ModelParameterGrid(param_grid)) == (
    ...    [{'a': 1, 'b': True}, {'a': 1, 'b': False},
    ...     {'a': 2, 'b': True}, {'a': 2, 'b': False}])
    True
    >>> grid = [{'kernel': ['linear']}, {'kernel': ['rbf'], 'gamma': [1, 10]}]
    >>> list(ModelParameterGrid(param_grid)) == [('model2', {'a': 0}),
    ...                                          ('model1', {'a': 1, 'b': True}),
    ...                                          ('model1', {'a': 1, 'b': False}),
    ...                                          ('model1', {'a': 2, 'b': True}),
    ...                                          ('model1', {'a': 2, 'b': False})]
    True
    >>> ModelParameterGrid(param_grid)[1] == ('model1', {'a': 1, 'b': True})
    True


    See also
    --------
    :class:`ModelAndGridSearchCV`:
        Uses :class:`ModelParameterGrid` to perform a full parallelized parameter
        search.

    """

    def __init__(self, param_grid):
        if isinstance(param_grid, Mapping):
            # wrap dictionary in a singleton list to support either dict
            # or list of dicts
            param_grid = [param_grid]
        self.param_grid = param_grid

    def __iter__(self):
        """Iterate over the points in the grid.
        Returns
        -------
        params : iterator over dict of string to any
            Yields dictionaries mapping each estimator parameter to one of its
            allowed values.
        """
        for p in self.param_grid:
            # Always sort the keys of a dictionary, for reproducibility
            items = list(p.items())
            if not items:
                yield {}
            else:
                for estimator, grid_list in items:
                    for grid in grid_list:
                        grid_points = sorted(list(grid.items()))
                        keys, values = zip(*grid_points)
                        for v in product(*values):
                            params = dict(zip(keys, v))
                            yield (estimator, params)

    def __len__(self):
        """Number of points on the grid."""
        # Product function that can handle iterables (np.product can't).
        return sum(_len(points) for p in self.param_grid for estim, points in list(p.items()))

    def __getitem__(self, ind):
        """Get the parameters that would be ``ind``th in iteration
        Parameters
        ----------
        ind : int
            The iteration index
        Returns
        -------
        params : dict of string to any
            Equal to list(self)[ind]
        """
        # This is used to make discrete sampling without replacement memory
        # efficient.
        for sub_grid in self.param_grid:
            for estimator, e_grid_list in list(sub_grid.items()):
                for e_grid in e_grid_list:
                    if not e_grid:
                        if ind == 0:
                            return (estimator, {})
                        else:
                            ind -= 1
                            continue

                    # Reverse so most frequent cycling parameter comes first
                    keys, values_lists = zip(*sorted(e_grid.items())[::-1])
                    sizes = [len(v_list) for v_list in values_lists]
                    total = np.product(sizes)

                    if ind >= total:
                        # Try the next grid
                        ind -= total
                    else:
                        out = {}
                        for key, v_list, n in zip(keys, values_lists, sizes):
                            ind, offset = divmod(ind, n)
                            out[key] = v_list[offset]
                        return (estimator, out)

        raise IndexError('ModelParameterGrid index out of range')

class RobustGridSearchCV(GridSearchCV):
    def _fit(self, X, y, groups, parameter_iterable):
        """Actual fitting,  performing the search over parameters."""

        estimator = self.estimator
        cv = check_cv(self.cv, y, classifier=is_classifier(estimator))
        self.scorer_ = check_scoring(self.estimator, scoring=self.scoring)

        X, y, groups = indexable(X, y, groups)
        n_splits = cv.get_n_splits(X, y, groups)
        if self.verbose > 0 and isinstance(parameter_iterable, Sized):
            n_candidates = len(parameter_iterable)
            print("Fitting {0} folds for each of {1} candidates, totalling"
                  " {2} fits".format(n_splits, n_candidates,
                                     n_candidates * n_splits))

        base_estimator = clone(self.estimator)
        pre_dispatch = self.pre_dispatch

        cv_iter = list(cv.split(X, y, groups))
        out = Parallel(
            n_jobs=self.n_jobs, verbose=self.verbose,
            pre_dispatch=pre_dispatch
        )(delayed(_robust_fit_and_score)(clone(base_estimator), X, y, self.scorer_,
                                  train, test, self.verbose, parameters,
                                  fit_params=self.fit_params,
                                  return_train_score=self.return_train_score,
                                  return_n_test_samples=True,
                                  return_times=True, return_parameters=True,
                                  error_score=self.error_score)
          for parameters in parameter_iterable
          for train, test in cv_iter)

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
            best_estimator = clone(base_estimator).set_params(
                **best_parameters)
            if y is not None:
                best_estimator.fit(X, y, **self.fit_params)
            else:
                best_estimator.fit(X, **self.fit_params)
            self.best_estimator_ = best_estimator
        return self

class ModelAndGridSearchCV(BaseSearchCV):
    """
    Adds model selection to the GridSearchCV
    """

    def __init__(self, param_grid, scoring=None, fit_params=None,
                 n_jobs=1, iid=True, refit=True, cv=None, verbose=0,
                 pre_dispatch='2*n_jobs', error_score='raise',
                 return_train_score=True):
        super(ModelAndGridSearchCV, self).__init__(
            estimator=None, scoring=scoring, fit_params=fit_params,
            n_jobs=n_jobs, iid=iid, refit=refit, cv=cv, verbose=verbose,
            pre_dispatch=pre_dispatch, error_score=error_score,
            return_train_score=return_train_score)
        self.param_grid = param_grid
        self.best_model_ = None
        # _check_param_grid(param_grid)

    def fit(self, X, y=None, groups=None):
        """
        Run fit with all sets of parameters.
        """
        return self._fit(X, y, groups, ModelParameterGrid(self.param_grid))

    def _fit(self, X, y, groups, parameter_iterable):
        """Actual fitting,  performing the search over parameters."""
        X, y, groups = indexable(X, y, groups)

        cv = check_cv(self.cv, y, classifier=True)
        n_splits = cv.get_n_splits(X, y, groups)

        if self.verbose > 0 and isinstance(parameter_iterable, Sized):
            n_candidates = len(parameter_iterable)
            LOG.info("Fitting %d folds for each of %d candidates, totalling"
                     " %d fits", n_splits, n_candidates, n_candidates * n_splits)

        pre_dispatch = self.pre_dispatch

        cv_iter = list(cv.split(X, y, groups))
        out = Parallel(
            n_jobs=self.n_jobs, verbose=self.verbose,
            pre_dispatch=pre_dispatch
        )(delayed(_model_fit_and_score)(
            estimator, X, y, self.scoring, train, test, self.verbose, parameters,
            fit_params=self.fit_params,
            return_train_score=self.return_train_score,
            return_n_test_samples=True,
            return_times=True, return_parameters=True,
            error_score=self.error_score)
          for estimator, parameters in parameter_iterable
          for train, test in cv_iter)

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
        best_parameters = candidate_params[best_index][1]

        # Use one MaskedArray and mask all the places where the param is not
        # applicable for that candidate. Use defaultdict as each candidate may
        # not contain all the params
        param_results = defaultdict(partial(MaskedArray,
                                            np.empty(n_candidates,),
                                            mask=True,
                                            dtype=object))
        for cand_i, params in enumerate(candidate_params):
            _, param_values = params
            for name, value in param_values.items():
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
        self.best_model_ = candidate_params[best_index]

        if self.refit:
            # build best estimator and fit
            best_estimator = _clf_build(self.best_model_[0])
            best_estimator.set_params(**best_parameters)
            if y is not None:
                best_estimator.fit(X, y, **self.fit_params)
            else:
                best_estimator.fit(X, **self.fit_params)
            self.best_estimator_ = best_estimator
        return self

def _model_fit_and_score(
    estimator_str, X, y, scorer, train, test, verbose,
    parameters, fit_params, return_train_score=False,
    return_parameters=False, return_n_test_samples=False,
    return_times=False, error_score='raise'):
    """

    """
    if verbose > 1:
        msg = '[CV model=%s]' % estimator_str.upper()
        if parameters is not None:
            msg += ' %s' % (', '.join('%s=%s' % (k, v)
                            for k, v in parameters.items()))
        LOG.info("%s %s", msg, (89 - len(msg)) * '.')

    estimator = _clf_build(estimator_str)

    # Adjust length of sample weights
    fit_params = fit_params if fit_params is not None else {}
    fit_params = dict([(k, _index_param_value(X, v, train))
                      for k, v in fit_params.items()])

    if parameters is not None:
        estimator.set_params(**parameters)

    start_time = time.time()

    X_train, y_train = _safe_split(estimator, X, y, train)
    X_test, y_test = _safe_split(estimator, X, y, test, train)

    try:
        if y_train is None:
            estimator.fit(X_train, **fit_params)
        else:
            estimator.fit(X_train, y_train, **fit_params)

    except Exception as e:
        # Note fit time as time until error
        fit_time = time.time() - start_time
        score_time = 0.0
        if error_score == 'raise':
            raise
        elif isinstance(error_score, numbers.Number):
            test_score = error_score
            if return_train_score:
                train_score = error_score
            warnings.warn("Classifier fit failed. The score on this train-test"
                          " partition for these parameters will be set to %f. "
                          "Details: \n%r" % (error_score, e), FitFailedWarning)
        else:
            raise ValueError("error_score must be the string 'raise' or a"
                             " numeric value. (Hint: if using 'raise', please"
                             " make sure that it has been spelled correctly.)")

    else:
        fit_time = time.time() - start_time
        scorer = check_scoring(estimator, scoring=scorer)
        test_score = _score(estimator, X_test, y_test, scorer)
        score_time = time.time() - start_time - fit_time
        if return_train_score:
            train_score = _score(estimator, X_train, y_train, scorer)

    if verbose > 2:
        msg += ", score=%f" % test_score
    if verbose > 1:
        total_time = score_time + fit_time
        end_msg = "%s, total=%s" % (msg, logger.short_format_time(total_time))
        LOG.info(end_msg)

    ret = [train_score, test_score] if return_train_score else [test_score]

    if return_n_test_samples:
        ret.append(_num_samples(X_test))
    if return_times:
        ret.extend([fit_time, score_time])
    if return_parameters:
        ret.append((estimator_str, parameters))
    return ret


def nested_fit_and_score(
        estimator, X, y, scorer, train, test, verbose=1,
        parameters=None, fit_params=None, return_train_score=False,
        return_times=False, error_score='raise'):
    """

    """
    from sklearn.externals.joblib.logger import short_format_time

    # Adjust length of sample weights
    fit_params = fit_params if fit_params is not None else {}
    fit_params = dict([(k, _index_param_value(X, v, train))
                      for k, v in fit_params.items()])

    if parameters is not None:
        estimator.set_params(**parameters)

    start_time = time.time()

    X_train, y_train = _safe_split(estimator, X, y, train)
    X_test, y_test = _safe_split(estimator, X, y, test, train)

    if verbose > 1:
        LOG.info('CV iteration: Xtrain=%d, Ytrain=%d/%d -- Xtest=%d, Ytest=%d/%d.',
                 len(X_train), len(X_train) - sum(y_train), sum(y_train),
                 len(X_test), len(X_test) - sum(y_test), sum(y_test))

    try:
        if y_train is None:
            estimator.fit(X_train, **fit_params)
        else:
            estimator.fit(X_train, y_train, **fit_params)

    except Exception as e:
        # Note fit time as time until error
        fit_time = time.time() - start_time
        score_time = 0.0
        if error_score == 'raise':
            raise
        elif isinstance(error_score, numbers.Number):
            test_score = error_score
            if return_train_score:
                train_score = error_score
            LOG.warn("Classifier fit failed. The score on this train-test"
                     " partition for these parameters will be set to %f. "
                     "Details: \n%r", error_score, e)
        else:
            raise ValueError("error_score must be the string 'raise' or a"
                             " numeric value. (Hint: if using 'raise', please"
                             " make sure that it has been spelled correctly.)")

    else:
        fit_time = time.time() - start_time

        test_score = None
        score_time = 0.0
        if len(set(y_test)) > 1:
            test_score = _score(estimator, X_test, y_test, scorer)
            score_time = time.time() - start_time - fit_time
        else:
            LOG.warn('Test set has no positive labels, scoring has been skipped '
                     'in this loop.')

        if return_train_score:
            train_score = _score(estimator, X_train, y_train, scorer)

        acc_score = _score(estimator, X_test, y_test,
                           check_scoring(estimator, scoring='accuracy'))

    if verbose > 0:
        total_time = score_time + fit_time
        if test_score is not None:
            LOG.info('Iteration took %s, score=%f, accuracy=%f.',
                     short_format_time(total_time), test_score, acc_score)
        else:
            LOG.info('Iteration took %s, score=None, accuracy=%f.',
                     short_format_time(total_time), acc_score)

    ret = {
        'test': {'roc_auc': test_score, 'accuracy': acc_score}
    }

    if return_train_score:
        ret['train'] = {'roc_auc': train_score}

    if return_times:
        ret['times'] = [fit_time, score_time]

    return ret, estimator

def _robust_fit_and_score(estimator, X, y, scorer, train, test, verbose,
                   parameters, fit_params, return_train_score=False,
                   return_parameters=False, return_n_test_samples=False,
                   return_times=False, error_score=0.5):
    """

    """
    if verbose > 1:
        if parameters is None:
            msg = ''
        else:
            msg = '%s' % (', '.join('%s=%s' % (k, v)
                          for k, v in parameters.items()))
        print("[CV] %s %s" % (msg, (64 - len(msg)) * '.'))

    # Adjust length of sample weights
    fit_params = fit_params if fit_params is not None else {}
    fit_params = dict([(k, _index_param_value(X, v, train))
                      for k, v in fit_params.items()])

    if parameters is not None:
        estimator.set_params(**parameters)

    start_time = time.time()

    X_train, y_train = _safe_split(estimator, X, y, train)
    X_test, y_test = _safe_split(estimator, X, y, test, train)

    try:
        if y_train is None:
            estimator.fit(X_train, **fit_params)
        else:
            estimator.fit(X_train, y_train, **fit_params)

    except Exception as e:
        # Note fit time as time until error
        fit_time = time.time() - start_time
        score_time = 0.0
        if error_score == 'raise':
            raise
        elif isinstance(error_score, numbers.Number):
            test_score = error_score
            if return_train_score:
                train_score = error_score
            warnings.warn("Classifier fit failed. The score on this train-test"
                          " partition for these parameters will be set to %f. "
                          "Details: \n%r" % (error_score, e), FitFailedWarning)
        else:
            raise ValueError("error_score must be the string 'raise' or a"
                             " numeric value. (Hint: if using 'raise', please"
                             " make sure that it has been spelled correctly.)")

    else:
        fit_time = time.time() - start_time
        if len(set(y_test)) == 1:
            test_score = 0.5
        else:
            test_score = _score(estimator, X_test, y_test, scorer)
        score_time = time.time() - start_time - fit_time
        if return_train_score:
            train_score = _score(estimator, X_train, y_train, scorer)

    if verbose > 2:
        msg += ", score=%f" % test_score
    if verbose > 1:
        total_time = score_time + fit_time
        end_msg = "%s, total=%s" % (msg, logger.short_format_time(total_time))
        print("[CV] %s %s" % ((64 - len(end_msg)) * '.', end_msg))

    ret = [train_score, test_score] if return_train_score else [test_score]

    if return_n_test_samples:
        ret.append(_num_samples(X_test))
    if return_times:
        ret.extend([fit_time, score_time])
    if return_parameters:
        ret.append(parameters)
    return ret

def _clf_build(clf_type):
    from sklearn import svm
    from sklearn.ensemble import RandomForestClassifier as RFC
    if clf_type == 'svc_linear':
        return svm.LinearSVC(C=1)
    elif clf_type == 'svc_rbf':
        return svm.SVC(C=1)
    elif clf_type == 'rfc':
        return RFC()
