#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: oesteban
# @Date:   2017-06-21 16:44:27


import warnings
import numbers
import time

import numpy as np

# import scipy.sparse as sp

from sklearn.base import is_classifier, clone
from sklearn.utils import indexable, check_random_state, safe_indexing
from sklearn.utils.validation import _num_samples
from sklearn.utils.metaestimators import _safe_split
from sklearn.externals.joblib import Parallel, delayed, logger
from sklearn.metrics.scorer import check_scoring
from sklearn.exceptions import FitFailedWarning
from sklearn.model_selection._split import check_cv
from sklearn.model_selection._validation import _index_param_value

# from sklearn.preprocessing import LabelEncoder

import logging

LOG = logging.getLogger("mriqc.classifier")


def cross_val_score(
    estimator,
    X,
    y=None,
    groups=None,
    scoring=None,
    cv=None,
    n_jobs=1,
    verbose=0,
    fit_params=None,
    pre_dispatch="2*n_jobs",
):
    """
    Evaluate a score by cross-validation
    """
    if not isinstance(scoring, (list, tuple)):
        scoring = [scoring]

    X, y, groups = indexable(X, y, groups)

    cv = check_cv(cv, y, classifier=is_classifier(estimator))
    splits = list(cv.split(X, y, groups))
    scorer = [check_scoring(estimator, scoring=s) for s in scoring]
    # We clone the estimator to make sure that all the folds are
    # independent, and that it is pickle-able.
    parallel = Parallel(n_jobs=n_jobs, verbose=verbose, pre_dispatch=pre_dispatch)
    scores = parallel(
        delayed(_fit_and_score)(
            clone(estimator), X, y, scorer, train, test, verbose, None, fit_params
        )
        for train, test in splits
    )

    group_order = []
    if hasattr(cv, "groups"):
        group_order = [np.array(cv.groups)[test].tolist()[0] for _, test in splits]
    return np.squeeze(np.array(scores)), group_order


def _fit_and_score(
    estimator,
    X,
    y,
    scorer,
    train,
    test,
    verbose,
    parameters,
    fit_params,
    return_train_score=False,
    return_parameters=False,
    return_n_test_samples=False,
    return_times=False,
    error_score="raise",
):
    """
    Fit estimator and compute scores for a given dataset split.
    """
    if verbose > 1:
        if parameters is None:
            msg = ""
        else:
            msg = "%s" % (", ".join("%s=%s" % (k, v) for k, v in parameters.items()))
        LOG.info("[CV] %s %s", msg, (64 - len(msg)) * ".")

    # Adjust length of sample weights
    fit_params = fit_params if fit_params is not None else {}
    fit_params = dict(
        [(k, _index_param_value(X, v, train)) for k, v in fit_params.items()]
    )

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
        if error_score == "raise":
            raise
        elif isinstance(error_score, numbers.Number):
            test_score = error_score
            if return_train_score:
                train_score = error_score
            warnings.warn(
                "Classifier fit failed. The score on this train-test"
                " partition for these parameters will be set to %f. "
                "Details: \n%r" % (error_score, e),
                FitFailedWarning,
            )
        else:
            raise ValueError(
                "error_score must be the string 'raise' or a"
                " numeric value. (Hint: if using 'raise', please"
                " make sure that it has been spelled correctly.)"
            )

    else:
        fit_time = time.time() - start_time
        test_score = [_score(estimator, X_test, y_test, s) for s in scorer]
        score_time = time.time() - start_time - fit_time
        if return_train_score:
            train_score = [_score(estimator, X_train, y_train, s) for s in scorer]

    if verbose > 2:
        msg += ", score=".join(("%f" % ts for ts in test_score))
    if verbose > 1:
        total_time = score_time + fit_time
        end_msg = "%s, total=%s" % (msg, logger.short_format_time(total_time))
        LOG.info("[CV] %s %s", (64 - len(end_msg)) * ".", end_msg)

    ret = [train_score, test_score] if return_train_score else [test_score]

    if return_n_test_samples:
        ret.append(_num_samples(X_test))
    if return_times:
        ret.extend([fit_time, score_time])
    if return_parameters:
        ret.append(parameters)
    return ret


def _score(estimator, X_test, y_test, scorer):
    """Compute the score of an estimator on a given test set."""
    if y_test is None:
        score = scorer(estimator, X_test)
    else:
        score = scorer(estimator, X_test, y_test)
    if hasattr(score, "item"):
        try:
            # e.g. unwrap memmapped scalars
            score = score.item()
        except ValueError:
            # non-scalar?
            pass
    if not isinstance(score, numbers.Number):
        raise ValueError(
            "scoring must return a number, got %s (%s) instead."
            % (str(score), type(score))
        )
    return score


def permutation_test_score(
    estimator,
    X,
    y,
    groups=None,
    cv=None,
    n_permutations=100,
    n_jobs=1,
    random_state=0,
    verbose=0,
    scoring=None,
):
    """
    Evaluate the significance of a cross-validated score with permutations,
    as in test 1 of [Ojala2010]_.

    A modification of original sklearn's permutation test score function
    to evaluate p-value outside this function, so that the score can be
    reused from outside.


    .. [Ojala2010] Ojala and Garriga. Permutation Tests for Studying Classifier
                   Performance.  The Journal of Machine Learning Research (2010)
                   vol. 11

    """
    X, y, groups = indexable(X, y, groups)

    cv = check_cv(cv, y, classifier=is_classifier(estimator))
    scorer = check_scoring(estimator, scoring=scoring)
    random_state = check_random_state(random_state)

    # We clone the estimator to make sure that all the folds are
    # independent, and that it is pickle-able.
    permutation_scores = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(_permutation_test_score)(
            clone(estimator), X, _shuffle(y, groups, random_state), groups, cv, scorer
        )
        for _ in range(n_permutations)
    )
    permutation_scores = np.array(permutation_scores)
    return permutation_scores


def _permutation_test_score(estimator, X, y, groups, cv, scorer):
    """Auxiliary function for permutation_test_score"""
    avg_score = []
    for train, test in cv.split(X, y, groups):
        X_train, y_train = _safe_split(estimator, X, y, train)
        X_test, y_test = _safe_split(estimator, X, y, test, train)
        estimator.fit(X_train, y_train)
        avg_score.append(scorer(estimator, X_test, y_test))
    return np.mean(avg_score)


def _shuffle(y, groups, random_state):
    """Return a shuffled copy of y eventually shuffle among same groups."""
    if groups is None:
        indices = random_state.permutation(len(y))
    else:
        indices = np.arange(len(groups))
        for group in np.unique(groups):
            this_mask = groups == group
            indices[this_mask] = random_state.permutation(indices[this_mask])
    return safe_indexing(y, indices)
