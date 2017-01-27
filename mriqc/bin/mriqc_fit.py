#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: oesteban
# @Date:   2015-11-19 16:44:27
# @Last Modified by:   oesteban
# @Last Modified time: 2017-01-27 10:47:21

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
    from mriqc.classifier.cv import NestedCVHelper
    from mriqc import logging, LOG_FORMAT

    warnings.showwarning = warn_redirect

    parser = ArgumentParser(description='MRIQC Nested cross-validation evaluation',
                            formatter_class=RawTextHelpFormatter)
    parser.add_argument('training_data', help='input data')
    parser.add_argument('training_labels', help='input data')

    g_input = parser.add_argument_group('Inputs')
    g_input.add_argument('-P', '--parameters', action='store',
                         default=pkgrf('mriqc', 'data/grid_nested_cv.yml'))

    g_input.add_argument('--cv-inner', action='store', default=10,
                         help='inner loop of cross-validation')
    g_input.add_argument('--cv-outer', action='store', default='loso',
                         help='outer loop of cross-validation')

    g_input.add_argument('--log-file', action='store', help='write log to this file')
    g_input.add_argument('--log-level', action='store', default='INFO',
                         choices=['CRITICAL', 'ERROR', 'WARN', 'INFO', 'DEBUG'])

    g_input.add_argument('-o', '--output-file', action='store', default='cv_inner_loop.csv',
                         help='the output table with cross validated scores')
    g_input.add_argument('-O', '--output-outer-cv', action='store', default='cv_outer_loop.csv',
                         help='the output table with cross validated scores')

    g_input.add_argument('--njobs', action='store', default=-1, type=int,
                         help='number of jobs')
    g_input.add_argument('--task-id', action='store')


    opts = parser.parse_args()

    logger = logging.getLogger()
    if opts.log_file is not None:
        fhl = logging.FileHandler(opts.log_file)
        fhl.setFormatter(fmt=logging.Formatter(LOG_FORMAT))
        logger.addHandler(fhl)
    logger.setLevel(opts.log_level)

    parameters = None
    if opts.parameters is not None:
        with open(opts.parameters) as paramfile:
            parameters = yaml.load(paramfile)

    cvhelper = NestedCVHelper(opts.training_data, opts.training_labels,
                              n_jobs=opts.njobs, param=parameters,
                              task_id=opts.task_id)

    cvhelper.cv_inner = read_cv(opts.cv_inner)
    cvhelper.cv_outer = read_cv(opts.cv_outer)

    # Run inner loop before setting held-out data, for hygene
    cvhelper.fit()
    with open(opts.output_file, 'a' if PY3 else 'ab') as outfile:
        flock(outfile, LOCK_EX)
        save_headers = op.getsize(opts.output_file) == 0
        cvhelper.get_inner_cv_scores().to_csv(
            outfile, index=False, header=save_headers)
        flock(outfile, LOCK_UN)

    with open(opts.output_outer_cv, 'a' if PY3 else 'ab') as outfile:
        flock(outfile, LOCK_EX)
        save_headers = op.getsize(opts.output_outer_cv) == 0
        cvhelper.get_outer_cv_scores().to_csv(
            outfile, index=False, header=save_headers)
        flock(outfile, LOCK_UN)

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
