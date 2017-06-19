#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: oesteban
# @Date:   2015-11-19 16:44:27
"""
mriqc_fit command line interface definition

"""
from __future__ import absolute_import, division, print_function, unicode_literals
from sys import version_info
import warnings
import matplotlib
matplotlib.use('Agg')

PY3 = version_info[0] > 2

try:
    from sklearn.metrics.base import UndefinedMetricWarning
except ImportError:
    from sklearn.exceptions import UndefinedMetricWarning

warnings.simplefilter("once", UndefinedMetricWarning)

cached_warnings = []


def warn_redirect(message, category, filename, lineno, file=None, line=None):
    from .. import logging
    log = logging.getLogger('mriqc.warnings')

    if category not in cached_warnings:
        log.debug('captured warning (%s): %s', category, message)
        cached_warnings.append(category)


def main():
    """Entry point"""
    import yaml
    from io import open
    from os.path import isfile
    from argparse import ArgumentParser
    from argparse import RawTextHelpFormatter
    from pkg_resources import resource_filename as pkgrf
    from .. import logging, LOG_FORMAT
    from ..classifier.helper import CVHelper

    warnings.showwarning = warn_redirect

    parser = ArgumentParser(description='MRIQC model selection and held-out evaluation',
                            formatter_class=RawTextHelpFormatter)

    g_clf = parser.add_mutually_exclusive_group()
    g_clf.add_argument('--train', nargs=2, help='training data tables, X and Y')
    g_clf.add_argument('--load-classifier', nargs="?", type=str, default='',
                       help='load pickled classifier in')

    parser.add_argument('--test-data', help='test data')
    parser.add_argument('--test-labels', help='test labels')

    parser.add_argument('--train-balanced-leaveout', action='store_true', default=False,
                        help='leave out a balanced, random, sample of training examples')
    parser.add_argument('--multiclass', '--ms', action='store_true', default=False,
                        help='do not binarize labels')

    parser.add_argument('-X', '--evaluation-data', help='classify this CSV table of IQMs')

    g_input = parser.add_argument_group('Inputs')
    g_input.add_argument('-P', '--parameters', action='store',
                         default=pkgrf('mriqc', 'data/classifier_settings.yml'))
    g_input.add_argument('-M', '--model', action='store', choices=['rfc', 'xgb'],
                         default='rfc', help='model')

    g_input.add_argument('-S', '--scorer', action='store', default='roc_auc')
    g_input.add_argument('--cv', action='store', default='loso',
                         choices=['kfold', 'loso', 'balanced-kfold', 'batch'])
    g_input.add_argument('--debug', action='store_true', default=False)

    g_input.add_argument('--log-file', action='store', help='write log to this file')
    g_input.add_argument("-v", "--verbose", dest="verbose_count",
                         action="count", default=0,
                         help="increases log verbosity for each occurence.")
    g_input.add_argument('--njobs', action='store', default=-1, type=int,
                         help='number of jobs')

    g_input.add_argument('-o', '--output', action='store', default='predicted_qa.csv',
                         help='file containing the labels assigned by the classifier')

    g_input.add_argument('-t', '--threshold', action='store', default=0.5, type=float,
                         help='decision threshold of the classifier')
    opts = parser.parse_args()

    log_level = int(max(3 - opts.verbose_count, 0) * 10)
    if opts.verbose_count > 1:
        log_level = int(max(25 - 5 * opts.verbose_count, 1))

    log = logging.getLogger('mriqc.classifier')
    log.setLevel(log_level)

    if opts.log_file is not None:
        fhl = logging.FileHandler(opts.log_file)
        fhl.setFormatter(fmt=logging.Formatter(LOG_FORMAT))
        fhl.setLevel(log_level)
        log.addHandler(fhl)

    parameters = None
    if opts.parameters is not None:
        with open(opts.parameters) as paramfile:
            parameters = yaml.load(paramfile)

    clf_loaded = False
    if opts.train is not None:
        train_exists = [isfile(fname) for fname in opts.train]
        if len(train_exists) > 0 and not all(train_exists):
            errors = ['file "%s" not found' % fname
                      for fexists, fname in zip(train_exists, opts.train)
                      if not fexists]
            raise RuntimeError('Errors (%d) loading training set: %s.' % (
                len(errors), ', '.join(errors)))

        # Initialize model selection helper
        cvhelper = CVHelper(
            X=opts.train[0],
            Y=opts.train[1],
            n_jobs=opts.njobs,
            param=parameters,
            scorer=opts.scorer,
            b_leaveout=opts.train_balanced_leaveout,
            multiclass=opts.multiclass,
            verbosity=opts.verbose_count,
            split=opts.cv,
            model=opts.model,
            debug=opts.debug
        )

        if opts.cv == 'batch':
            cvhelper.setXtest(opts.test_data, opts.test_labels)

        # Perform model selection before setting held-out data, for hygene
        cvhelper.fit()

        # Pickle if required
        cvhelper.save(suffix='data-train_estimator')

    # If no training set is given, need a classifier
    else:
        load_classifier = opts.load_classifier
        if load_classifier is None:
            load_classifier = pkgrf('mriqc', 'data/rfc-nzs-full-1.0.pklz')

        if not isfile(load_classifier):
            msg = 'was not provided'
            if load_classifier != '':
                msg = '("%s") was not found' % load_classifier
            raise RuntimeError(
                'No training samples were given, and the --load-classifier '
                'option %s.' % msg)

        cvhelper = CVHelper(load_clf=load_classifier, n_jobs=opts.njobs)
        clf_loaded = True

    if opts.test_data and opts.test_labels and opts.cv != 'batch':
        # Set held-out data
        cvhelper.setXtest(opts.test_data, opts.test_labels)
        # Evaluate
        cvhelper.evaluate(matrix=True, scoring=[opts.scorer, 'accuracy'],
                          save_pred=True)

        # Pickle if required
        if not clf_loaded:
            cvhelper.fit_full()
            cvhelper.save(suffix='data-all_estimator')

    if opts.evaluation_data:
        cvhelper.predict_dataset(opts.evaluation_data, out_file=opts.output, thres=opts.threshold)


if __name__ == '__main__':
    main()
