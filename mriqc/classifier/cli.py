#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: oesteban
# @Date:   2015-11-19 16:44:27
# @Last Modified by:   oesteban
# @Last Modified time: 2016-05-26 16:56:00

"""
MRIQC Cross-validation

"""
from __future__ import absolute_import, division, print_function, unicode_literals

def main():
    """Entry point"""
    import json
    from io import open
    from argparse import ArgumentParser
    from argparse import RawTextHelpFormatter
    from .cv import CVHelper

    parser = ArgumentParser(description='MRIQC Cross-validation',
                            formatter_class=RawTextHelpFormatter)
    parser.add_argument('training_data', help='input data')
    parser.add_argument('training_labels', help='input data')

    parser.add_argument('--test-data', help='test data')
    parser.add_argument('--test-labels', help='test labels')

    g_input = parser.add_argument_group('Inputs')
    g_input.add_argument('-P', '--parameters', action='store')
    g_input.add_argument('-C', '--classifier', action='store', nargs='*',
                         choices=['svc_linear', 'svc_rbf', 'rfc', 'all'],
                         default=['svc_rbf'])
    g_input.add_argument(
        '-S', '--score-types', action='store', nargs='*', default=['f1_weighted', 'accuracy'],
        choices=[
            'accuracy', 'adjusted_rand_score', 'average_precision', 'f1', 'f1_macro', 'f1_micro',
            'f1_samples', 'f1_weighted', 'log_loss', 'mean_absolute_error', 'mean_squared_error',
            'median_absolute_error', 'precision', 'precision_macro', 'precision_micro',
            'precision_samples', 'precision_weighted', 'r2', 'recall', 'recall_macro',
            'recall_micro', 'recall_samples', 'recall_weighted', 'roc_auc'])

    opts = parser.parse_args()

    parameters = None
    if opts.parameters is not None:
        with open(opts.parameters) as paramfile:
            parameters = json.load(paramfile)

    cvhelper = CVHelper(opts.training_data, opts.training_labels,
                        scores=opts.score_types, param=parameters)

    # Run inner loop before setting held-out data, for hygene
    cvhelper.inner_loop()

    if opts.test_data is not None:
        cvhelper.set_heldout_dataset(opts.test_data, opts.test_labels)


if __name__ == '__main__':
    main()
