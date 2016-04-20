#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: oesteban
# @Date:   2015-11-19 16:44:27
# @Last Modified by:   oesteban
# @Last Modified time: 2016-04-20 16:40:13

"""
MRIQC Plot script

"""
import os
import os.path as op
import collections
from multiprocessing import cpu_count

from argparse import ArgumentParser
from argparse import RawTextHelpFormatter

import glob
import json
import pandas as pd

from mriqc import __version__


def main():
    """Entry point"""
    parser = ArgumentParser(description='MRI Quality Control',
                            formatter_class=RawTextHelpFormatter)

    g_input = parser.add_argument_group('Inputs')
    g_input.add_argument('-d', '--data-type', action='store', choices=['anat', 'func'])
    g_input.add_argument('-v', '--version', action='store_true', default=False,
                         help='Show current mriqc version')

    g_input.add_argument('--nthreads', action='store', default=0,
                         type=int, help='number of threads')

    g_outputs = parser.add_argument_group('Outputs')
    g_outputs.add_argument('-o', '--output-dir', action='store')
    g_outputs.add_argument('-w', '--work-dir', action='store', default=op.join(os.getcwd(), 'work'))

    opts = parser.parse_args()
    if opts.version:
        print 'mriqc version ' + __version__
        exit(0)

    settings = {'output_dir': os.getcwd(),
                'nthreads': opts.nthreads}

    if opts.output_dir:
        settings['output_dir'] = op.abspath(opts.output_dir)

    if not op.exists(settings['output_dir']):
        os.makedirs(settings['output_dir'])

    settings['work_dir'] = op.abspath(opts.work_dir)
    if not op.exists(settings['work_dir']):
        raise RuntimeError('Work directory of a previous MRIQC run was not found.')

    datalist = []
    for jsonfile in glob.glob(op.join(settings['work_dir'], 'derivatives', '*.json')):
        datalist.append(_read_and_save(jsonfile))

    dataframe = pd.DataFrame(datalist)
    cols = dataframe.columns.tolist()  # pylint: disable=no-member

    for col in ['run_id', 'session_id', 'subject_id']:
        cols.remove(col)
        cols.insert(0, col)

    dataframe = dataframe.sort_values(by=['subject_id', 'session_id', 'run_id'])

    out_fname = op.join(settings['output_dir'], opts.data_type + 'MRIQC.csv')
    dataframe[cols].to_csv(out_fname, index=False)


def _read_and_save(in_file):
    with open(in_file, 'r') as jsondata:
        values = _flatten(json.load(jsondata))
        return values
    return None


def _flatten(in_dict, parent_key='', sep='_'):
    items = []
    for k, val in list(in_dict.items()):
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(val, collections.MutableMapping):
            items.extend(_flatten(val, new_key, sep=sep).items())
        else:
            items.append((new_key, val))
    return dict(items)


if __name__ == '__main__':
    main()
