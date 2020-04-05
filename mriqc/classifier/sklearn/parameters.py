#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: oesteban
# @Date:   2015-11-19 16:44:27

"""
Parameters grid
===============


Extends sklearn's ModelParameterGrid so the grid includes
different models.


"""

from itertools import product
from collections import Mapping
import operator
import numpy as np
from builtins import object, zip
from functools import partial, reduce


def _len(indict):
    product = partial(reduce, operator.mul)
    return sum(product(len(v) for v in p.values()) if p else 1 for p in indict)


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
        return sum(
            _len(points) for p in self.param_grid for estim, points in list(p.items())
        )

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

        raise IndexError("ModelParameterGrid index out of range")
