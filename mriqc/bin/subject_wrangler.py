#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: oesteban
# @Date:   2015-11-19 16:44:27
# @Last Modified by:   oesteban
# @Last Modified time: 2018-03-12 11:51:32

"""
BIDS-Apps subject wrangler

"""
from builtins import range  # pylint: disable=W0622

import os.path as op
import glob
from random import shuffle
# from lockfile import LockFile

from argparse import ArgumentParser
from argparse import RawTextHelpFormatter
from textwrap import dedent

from .. import __version__


def main():
    """Entry point"""
    parser = ArgumentParser(formatter_class=RawTextHelpFormatter, description=dedent("""\
BIDS-Apps participants wrangler tool
------------------------------------

This command arranges the participant labels in groups for computation, and checks that the \
requested participants have the corresponding folder in the bids_dir.\
"""))

    parser.add_argument('-v', '--version', action='version',
                        version='mriqc v{}'.format(__version__))

    parser.add_argument('bids_dir', action='store',
                        help='The directory with the input dataset '
                             'formatted according to the BIDS standard.')
    parser.add_argument('output_dir', action='store',
                        help='The directory where the output files '
                             'should be stored. If you are running group level analysis '
                             'this folder should be prepopulated with the results of the'
                             'participant level analysis.')
    parser.add_argument('--participant_label', '--subject_list', '-S', action='store',
                        help='The label(s) of the participant(s) that should be analyzed. '
                             'The label corresponds to sub-<participant_label> from the '
                             'BIDS spec (so it does not include "sub-"). If this parameter '
                             'is not provided all subjects should be analyzed. Multiple '
                             'participants can be specified with a space separated list.',
                        nargs="*")
    parser.add_argument('--group-size', default=1, action='store', type=int,
                        help='parallelize participants in groups')
    parser.add_argument('--no-randomize', default=False, action='store_true',
                        help='do not randomize participants list before grouping')
    parser.add_argument('--log-groups', default=False, action='store_true',
                        help='append logging output')
    parser.add_argument('--multiple-workdir', default=False, action='store_true',
                        help='split work directories by jobs')
    parser.add_argument('--bids-app-name', default='mriqc', action='store',
                        help='BIDS app to call')
    parser.add_argument('--args', default='', action='store', help='append arguments')

    opts = parser.parse_args()

    # Build settings dict
    bids_dir = op.abspath(opts.bids_dir)
    all_subjects = sorted([op.basename(subj)[4:]
                           for subj in glob.glob(op.join(bids_dir, 'sub-*'))])

    subject_list = opts.participant_label
    if subject_list is None or not subject_list:
        subject_list = all_subjects
    else:
        # remove sub- prefix, get unique
        for i, subj in enumerate(subject_list):
            subject_list[i] = subj[4:] if subj.startswith('sub-') else subj

        subject_list = sorted(list(set(subject_list)))

        if list(set(subject_list) - set(all_subjects)):
            non_exist = list(set(subject_list) - set(all_subjects))
            raise RuntimeError('Participant label(s) not found in the '
                               'BIDS root directory: {}'.format(' '.join(non_exist)))

    if not opts.no_randomize:
        shuffle(subject_list)

    gsize = opts.group_size

    if gsize < 0:
        raise RuntimeError('group size should be at least 0 '
                           '(all participants assigned to same group')
    if gsize == 0:
        gsize = len(subject_list)

    groups = [subject_list[i:i + gsize]
              for i in range(0, len(subject_list), gsize)]

    log_arg = ''.format
    if opts.log_groups:
        log_arg = '>> log/mriqc-{:04d}.log'.format

    cmdline = ('{exec} {bids_dir} {out_dir} participant --participant_label {labels}'
               '{work_dir} {arguments} {logfile}').format
    for i, part_group in enumerate(groups):
        workdir = ''
        if opts.multiple_workdir:
            workdir = ' -w work/sjob-{:04d}'.format(i)
        print(cmdline(**{
            'exec': opts.bids_app_name, 'bids_dir': bids_dir, 'out_dir': opts.output_dir,
            'labels': ' '.join(part_group), 'work_dir': workdir, 'arguments': opts.args,
            'logfile': log_arg(i)}))


if __name__ == '__main__':
    main()
