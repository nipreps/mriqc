#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: oesteban
# @Date:   2015-11-19 16:44:27
# @Last Modified by:   oesteban
# @Last Modified time: 2017-01-23 14:41:50

"""
mriqc_fit command line interface definition

"""
from __future__ import absolute_import, division, print_function, unicode_literals
from sys import version_info
import os.path as op
from fcntl import flock, LOCK_EX, LOCK_UN
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

    warnings.showwarning = warn_redirect

    parser = ArgumentParser(description='MRIQC model selection and held-out evaluation',
                            formatter_class=RawTextHelpFormatter)
    parser.add_argument('training_data', help='input data')
    parser.add_argument('training_labels', help='input data')

    parser.add_argument('--test-data', help='test data')
    parser.add_argument('--test-labels', help='test labels')

    g_input = parser.add_argument_group('Inputs')
    g_input.add_argument('-P', '--parameters', action='store',
                         default=pkgrf('mriqc', 'data/classifier_settings.yml'))

    g_input.add_argument('--load-classifier', action='store', help='load pickled classifier in')
    g_input.add_argument('--save-classifier', action='store', help='write pickled classifier out')

    g_input.add_argument('--log-file', action='store', help='write log to this file')
    g_input.add_argument('--log-level', action='store', default='INFO',
                         choices=['CRITICAL', 'ERROR', 'WARN', 'INFO', 'DEBUG'])
    g_input.add_argument('--njobs', action='store', default=-1, type=int,
                         help='number of jobs')


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

    if opts.load_classifier is None:
        # Initialize model selection helper
        cvhelper = CVHelper(opts.training_data, opts.training_labels, n_jobs=opts.njobs,
                            param=parameters)

        # Perform model selection before setting held-out data, for hygene
        cvhelper.fit()

        # Pickle if required
        if opts.save_classifier:
            cvhelper.save(opts.save_classifier)

    else:
        # cvhelper = CVHelper(opts.load_classifier, n_jobs=opts.njobs)
        raise NotImplementedError

    if opts.test_data and opts.test_labels:
        # Set held-out data
        cvhelper.setXtest(opts.test_data, opts.test_labels)
        # Evaluate
        cvhelper.evaluate()
        # Get score
        cvhelper.get_score()


if __name__ == '__main__':
    main()
