#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: oesteban
# @Date:   2016-03-16 11:28:27
# @Last Modified by:   oesteban
# @Last Modified time: 2016-03-16 15:12:13

"""
Compares pandas dataframes by columns

"""
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
    tstdf = pd.read_csv(opts.input_csv)
    refdf = pd.read_csv(opts.reference_csv)

    return np.all(tstdf == refdf)


if __name__ == '__main__':
    main()
