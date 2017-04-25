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
    from mriqc.classifier.data import read_iqms
    parser = ArgumentParser(description='compare two pandas dataframes',
                            formatter_class=RawTextHelpFormatter)
    g_input = parser.add_argument_group('Inputs')
    g_input.add_argument('-i', '--input-csv', action='store',
                         required=True, help='input data frame')
    g_input.add_argument('-r', '--reference-csv', action='store',
                         required=True, help='reference dataframe')

    opts = parser.parse_args()

    ref_df, ref_names, ref_bids = read_iqms(opts.reference_csv)
    tst_df, tst_names, tst_bids = read_iqms(opts.input_csv)

    if sorted(ref_bids) != sorted(tst_bids):
        sys.exit('Dataset has different BIDS bits w.r.t. reference')

    if sorted(ref_names) != sorted(tst_names):
        sys.exit('Output CSV file changed number of columns')

    ref_df = ref_df.sort_values(by=ref_bids)
    tst_df = tst_df.sort_values(by=tst_bids)

    if not np.all(ref_df[ref_names].values == tst_df[tst_names].values):
        sys.exit('Output CSV file changed one or more values')

    sys.exit(0)


if __name__ == '__main__':
    main()
