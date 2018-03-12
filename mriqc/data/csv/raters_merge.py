#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
# @Author: oesteban
# @Date:   2017-06-19 10:06:20
import pandas as pd


def get_parser():
    """Entry point"""
    from argparse import ArgumentParser
    from argparse import RawTextHelpFormatter
    parser = ArgumentParser(description='Merge ratings from two raters',
                            formatter_class=RawTextHelpFormatter)
    g_input = parser.add_argument_group('Inputs')
    g_input.add_argument('rater_1', action='store')
    g_input.add_argument('rater_2', action='store')
    g_input.add_argument('--mapping-file', action='store')

    g_outputs = parser.add_argument_group('Outputs')
    g_outputs.add_argument('-o', '--output', action='store', default='merged.csv')
    return parser


def main():
    opts = get_parser().parse_args()

    rater_1 = pd.read_csv(opts.rater_1)[['participant_id', 'check-1']]
    rater_2 = pd.read_csv(opts.rater_2)[['participant_id', 'check-1']]

    rater_1.columns = ['participant_id', 'rater_1']
    rater_2.columns = ['participant_id', 'rater_2']
    merged = pd.merge(rater_1, rater_2, on='participant_id', how='outer')

    idcol = 'participant_id'
    if opts.mapping_file:
        idcol = 'subject_id'
        name_mapping = pd.read_csv(
            opts.mapping_file, sep=' ', header=None, usecols=[0, 1])
        name_mapping.columns = ['subject_id', 'participant_id']
        name_mapping['participant_id'] = name_mapping.participant_id.astype(str) + '.gif'
        merged = pd.merge(name_mapping, merged, on='participant_id', how='outer')

    merged[[idcol, 'rater_1', 'rater_2']].sort_values(by=idcol).to_csv(opts.output, index=False)


if __name__ == '__main__':
    main()
