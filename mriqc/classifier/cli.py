#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: oesteban
# @Date:   2015-11-19 16:44:27
# @Last Modified by:   oesteban
# @Last Modified time: 2016-12-13 17:43:34

"""
MRIQC Cross-validation

"""
from __future__ import absolute_import, division, print_function, unicode_literals
import warnings


from sklearn.metrics.base import UndefinedMetricWarning
warnings.simplefilter("once", UndefinedMetricWarning)

cached_warnings = []
def warn_redirect(message, category, filename, lineno, file=None, line=None):
    from mriqc import logging
    LOG = logging.getLogger('mriqc.warnings')

    if category not in cached_warnings:
        LOG.debug('captured warning (%s): %s', category, message)
        cached_warnings.append(category)



def main():
    """Entry point"""
    import yaml
    from io import open
    from argparse import ArgumentParser
    from argparse import RawTextHelpFormatter
    from pkg_resources import resource_filename as pkgrf
    from .cv import CVHelper
    from mriqc import logging, LOG_FORMAT

    warnings.showwarning = warn_redirect

    parser = ArgumentParser(description='MRIQC Cross-validation',
                            formatter_class=RawTextHelpFormatter)
    parser.add_argument('training_data', help='input data')
    parser.add_argument('training_labels', help='input data')

    parser.add_argument('--test-data', help='test data')
    parser.add_argument('--test-labels', help='test labels')

    g_input = parser.add_argument_group('Inputs')
    g_input.add_argument('-P', '--parameters', action='store',
                         default=pkgrf('mriqc', 'data/grid_nested_cv.yml'))
    g_input.add_argument('-C', '--classifier', action='store', nargs='*',
                         choices=['svc_linear', 'svc_rbf', 'rfc', 'all'],
                         default=['svc_rbf'])

    g_input.add_argument('--cv-inner', action='store', default=10,
                         help='inner loop of cross-validation')
    g_input.add_argument('--cv-outer', action='store', default='loso',
                         help='outer loop of cross-validation')

    g_input.add_argument('--create-split', action='store_true', default=False,
                         help='create a data split for the validation set')

    g_input.add_argument('--nperm', action='store', default=5000, type=int,
                         help='number of permutations')

    g_input.add_argument(
        '-S', '--score-types', action='store', nargs='*', default=['accuracy'],
        choices=[
            'accuracy', 'adjusted_rand_score', 'average_precision', 'f1', 'f1_macro', 'f1_micro',
            'f1_samples', 'f1_weighted', 'log_loss', 'mean_absolute_error', 'mean_squared_error',
            'median_absolute_error', 'precision', 'precision_macro', 'precision_micro',
            'precision_samples', 'precision_weighted', 'r2', 'recall', 'recall_macro',
            'recall_micro', 'recall_samples', 'recall_weighted', 'roc_auc'])

    g_input.add_argument('--log-file', action='store', default='mriqcfit.log')
    g_input.add_argument('--log-level', action='store', default='INFO',
                         choices=['CRITICAL', 'ERROR', 'WARN', 'INFO', 'DEBUG'])

    opts = parser.parse_args()

    filelogger = logging.getLogger()
    fhl = logging.FileHandler(opts.log_file)
    fhl.setFormatter(fmt=logging.Formatter(LOG_FORMAT))
    filelogger.addHandler(fhl)
    filelogger.setLevel(opts.log_level)

    parameters = None
    if opts.parameters is not None:
        with open(opts.parameters) as paramfile:
            parameters = yaml.load(paramfile)

    cvhelper = CVHelper(opts.training_data, opts.training_labels,
                        scores=opts.score_types, param=parameters, n_perm=opts.nperm)

    cvhelper.cv_inner = read_cv(opts.cv_inner)
    cvhelper.cv_outer = read_cv(opts.cv_outer)

    # Run inner loop before setting held-out data, for hygene
    cvhelper.fit()


def read_cv(value):
    from numbers import Number

    try:
        value = int(value)
    except ValueError:
        pass

    if isinstance(value, Number):
        if value > 0:
            return {'type': 'kfold', 'n_splits': value}
        else:
            return None
    return {'type': 'loso'}


if __name__ == '__main__':
    main()
