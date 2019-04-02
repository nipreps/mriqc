#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: oesteban
# @Date:   2016-03-16 11:28:27
# @Last Modified by:   oesteban
# @Last Modified time: 2018-08-15 10:54:26

"""
Compares pandas dataframes by columns

"""

import sys
from pathlib import Path
from argparse import ArgumentParser, RawTextHelpFormatter
import numpy as np
import pandas as pd


def main():
    """Entry point"""
    from ..classifier.data import read_iqms
    parser = ArgumentParser(description='compare two pandas dataframes',
                            formatter_class=RawTextHelpFormatter)
    g_input = parser.add_argument_group('Inputs')
    g_input.add_argument('-i', '--input-csv', action='store', type=Path,
                         required=True, help='input data frame')
    g_input.add_argument('-r', '--reference-csv', action='store', type=Path,
                         required=True, help='reference dataframe')
    g_input.add_argument('--tolerance', type=float, default=1.e-5,
                         help='relative tolerance for comparison')

    opts = parser.parse_args()

    ref_df, ref_names, ref_bids = read_iqms(opts.reference_csv)
    tst_df, tst_names, tst_bids = read_iqms(opts.input_csv)

    ref_df.set_index(ref_bids)
    tst_df.set_index(tst_bids)

    if sorted(ref_bids) != sorted(tst_bids):
        sys.exit('Dataset has different BIDS bits w.r.t. reference')

    if sorted(ref_names) != sorted(tst_names):
        sys.exit('Output CSV file changed number of columns')

    ref_df = ref_df.sort_values(by=ref_bids)
    tst_df = tst_df.sort_values(by=tst_bids)

    if len(ref_df) != len(tst_df):
        print('Input datases have different lengths (input %d, reference %d).' % (
            len(ref_df), len(tst_df)))
        tst_rows = tst_df[tst_bids]
        ref_rows = ref_df[ref_bids]

        print(tst_rows.shape, ref_rows.shape)

        tst_keep = np.sum(tst_rows.isin(ref_rows).values.ravel().tolist())
        print(tst_keep)

    diff = ~np.isclose(ref_df[ref_names].values, tst_df[tst_names].values, rtol=opts.tolerance)
    if np.any(diff):
        # ne_stacked = pd.DataFrame(data=diff, columns=ref_names).stack()
        # ne_stacked = np.isclose(ref_df[ref_names], tst_df[ref_names]).stack()
        # changed = ne_stacked[ne_stacked]
        # changed.set_index(ref_bids)
        difference_locations = np.where(diff)
        changed_from = ref_df[ref_names].values[difference_locations]
        changed_to = tst_df[ref_names].values[difference_locations]
        cols = [ref_names[v] for v in difference_locations[1]]
        bids_df = ref_df.loc[difference_locations[0], ref_bids].reset_index()
        chng_df = pd.DataFrame({'iqm': cols, 'from': changed_from, 'to': changed_to})
        table = pd.concat([bids_df, chng_df], axis=1)
        print(table[ref_bids + ['iqm', 'from', 'to']].to_string(index=False))

        corr = pd.DataFrame()
        corr['iqms'] = ref_names
        corr['cc'] = [float(np.corrcoef(ref_df[[var]].values.ravel(),
                                        tst_df[[var]].values.ravel(),
                                        rowvar=False)[0, 1])
                      for var in ref_names]

        if np.any(corr.cc < 0.95):
            print('IQMs with Pearson correlation < 0.95:')
            print(corr[corr.cc < 0.95])
            sys.exit('Output CSV file changed one or more values')
        else:
            print('All IQMs show a Pearson correlation >= 0.95')

    sys.exit(0)


if __name__ == '__main__':
    main()
