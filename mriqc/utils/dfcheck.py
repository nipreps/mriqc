#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: oesteban
# @Date:   2016-03-16 11:28:27
# @Last Modified by:   oesteban
# @Last Modified time: 2016-03-16 15:12:13

"""
Compares pandas dataframes by columns

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import sys
from argparse import ArgumentParser, RawTextHelpFormatter
import numpy as np
import pandas as pd


def main():
    """Entry point"""
    parser = ArgumentParser(description='compare two pandas dataframes',
                            formatter_class=RawTextHelpFormatter)
    g_input = parser.add_argument_group('Inputs')
    g_input.add_argument('-i', '--input-csv', action='store',
                         required=True, help='input data frame')
    g_input.add_argument('-r', '--reference-csv', action='store',
                         required=True, help='reference dataframe')

    opts = parser.parse_args()
    tstdf = pd.read_csv(opts.input_csv).sort_values(['subject_id', 'session_id', 'run_id'],
                                                    ascending=[True, True, True])
    refdf = pd.read_csv(opts.reference_csv).sort_values(['subject_id', 'session_id', 'run_id'],
                                                        ascending=[True, True, True])

    refcolumns = refdf.columns.ravel().tolist()
    for col in refcolumns:
        if 'Unnamed' in col:
            refcolumns.remove(col)

    tstcolumns = tstdf.columns.ravel().tolist()
    for col in tstcolumns:
        if 'Unnamed' in col:
            tstcolumns.remove(col)

    if sorted(refcolumns) != sorted(tstcolumns):
        sys.exit('Output CSV file changed number of columns')

    if not np.all(refdf[refcolumns].values == tstdf[refcolumns].values):
        sys.exit('Output CSV file changed one or more values')

    sys.exit(0)


if __name__ == '__main__':
    main()
