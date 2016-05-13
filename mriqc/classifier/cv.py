#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: oesteban
# @Date:   2015-11-19 16:44:27
# @Last Modified by:   oesteban
# @Last Modified time: 2016-05-13 10:11:21

"""
MRIQC Cross-validation

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import os.path as op
from argparse import ArgumentParser
from argparse import RawTextHelpFormatter

import pandas as pd
from sklearn import svm
from sklearn.cross_validation import LeaveOneLabelOut

def main():
    """Entry point"""
    parser = ArgumentParser(description='MRI Quality Control',
                            formatter_class=RawTextHelpFormatter)

    g_input = parser.add_argument_group('Inputs')
    g_input.add_argument('-X', '--in-training', action='store',
                         required=True)
    g_input.add_argument('-y', '--in-training-labels', action='store',
                         required=True)

    g_outputs = parser.add_argument_group('Outputs')
    g_outputs.add_argument('-o', '--out-csv', action='store',
                           help='output CSV file combining X and y')
    opts = parser.parse_args()

    with open(opts.in_training, 'r') as fileX:
    	X_df = pd.read_csv(fileX).sort_values(by=['subject_id'])

    with open(opts.in_training_labels, 'r') as fileY:
    	y_df = pd.read_csv(fileY).sort_values(by=['subject_id'])

    # Remove columns that are not IQMs
    columns = list(X_df.columns.ravel())
    columns.remove('subject_id')
    columns.remove('session_id')
    columns.remove('run_id')

    # Remove failed cases from Y, append new columns to X
    y_df = y_df[y_df['subject_id'].isin(X_df.subject_id)]
    sites = list(y_df.site.values)
    X_df['rate'] = y_df.rate.values
    X_df['site'] = y_df.site.values

    if opts.out_csv is not None:
        with open(opts.out_csv, 'w') as outfile:
            X_df.to_csv(outfile, index=False)

	# Convert all samples to tuples
    X = [tuple(x) for x in X_df[columns].values]

    # lolo = LeaveOneLabelOut(sites)
    clf = svm.SVC()
    clf.fit(X, list(y_df.rate.values))

if __name__ == '__main__':
    main()
