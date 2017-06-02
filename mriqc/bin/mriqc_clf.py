#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: oesteban
# @Date:   2015-11-19 16:44:27
# @Last Modified by:   oesteban
# @Last Modified time: 2017-06-02 15:26:34

"""
mriqc_fit command line interface definition

"""
from __future__ import absolute_import, division, print_function, unicode_literals
from sys import version_info
import warnings

PY3 = version_info[0] > 2

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
    from mriqc.classifier.cv import CVHelper
    from mriqc import logging, LOG_FORMAT
    from os.path import isfile

    warnings.showwarning = warn_redirect

    parser = ArgumentParser(description='MRIQC model selection and held-out evaluation',
                            formatter_class=RawTextHelpFormatter)

    g_clf = parser.add_mutually_exclusive_group()
    g_clf.add_argument('--train', nargs=2, help='training data tables, X and Y')
    g_clf.add_argument('--load-classifier', nargs="?", type=str, default='',
                       help='load pickled classifier in')

    parser.add_argument('--test-data', help='test data')
    parser.add_argument('--test-labels', help='test labels')

    parser.add_argument('-X', '--evaluation-data', help='classify this CSV table of IQMs')

    g_input = parser.add_argument_group('Inputs')
    g_input.add_argument('-P', '--parameters', action='store',
                         default=pkgrf('mriqc', 'data/classifier_settings.yml'))

    g_input.add_argument('--save-classifier', action='store', help='write pickled classifier out')

    g_input.add_argument('--log-file', action='store', help='write log to this file')
    g_input.add_argument('--log-level', action='store', default='INFO',
                         choices=['CRITICAL', 'ERROR', 'WARN', 'INFO', 'DEBUG'])
    g_input.add_argument('--njobs', action='store', default=-1, type=int,
                         help='number of jobs')

    g_input.add_argument('-o', '--output', action='store', default='predicted_qa.csv',
                         help='file containing the labels assigned by the classifier')


    opts = parser.parse_args()

    if opts.log_file is not None:
        filelogger = logging.getLogger()
        fhl = logging.FileHandler(opts.log_file)
        fhl.setFormatter(fmt=logging.Formatter(LOG_FORMAT))
        filelogger.addHandler(fhl)
        filelogger.setLevel(opts.log_level)

    parameters = None
    if opts.parameters is not None:
        with open(opts.parameters) as paramfile:
            parameters = yaml.load(paramfile)

    if opts.train is not None:
        train_exists = [isfile(fname) for fname in opts.train]
        if len(train_exists) > 0 and not all(train_exists):
            errors = ['file "%s" not found' % fname
                      for fexists, fname in zip(train_exists, opts.train)
                      if not fexists]
            raise RuntimeError('Errors (%d) loading training set: %s.' % (
                len(errors), ', '.join(errors)))

        # Initialize model selection helper
        cvhelper = CVHelper(X=opts.train[0], Y=opts.train[1], n_jobs=opts.njobs,
                            param=parameters)

        # Perform model selection before setting held-out data, for hygene
        cvhelper.fit()

        # Pickle if required
        if opts.save_classifier:
            cvhelper.save(opts.save_classifier)

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

        cvhelper = CVHelper(load_clf=load_classifier, n_jobs=opts.njobs,
                            rate_label='rate')

    if opts.test_data and opts.test_labels:
        # Set held-out data
        cvhelper.setXtest(opts.test_data, opts.test_labels)
        # Evaluate
        print('roc_auc=%f, accuracy=%f' % (cvhelper.evaluate(scoring='roc_auc'),
                                           cvhelper.evaluate()))

    if opts.evaluation_data:
        cvhelper.predict_dataset(opts.evaluation_data, out_file=opts.output)


if __name__ == '__main__':
    main()
