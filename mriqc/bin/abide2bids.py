#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: oesteban
# @Date:   2016-03-16 11:28:27
# @Last Modified by:   oesteban
# @Last Modified time: 2018-03-12 11:45:42

"""
ABIDE2BIDS downloader tool

"""
from __future__ import absolute_import, division, print_function, unicode_literals

import os
import os.path as op
import errno
import shutil
import json
import subprocess as sp
import tempfile
from xml.etree import ElementTree as et
from multiprocessing import Pool
from argparse import ArgumentParser, RawTextHelpFormatter
import numpy as np


def main():
    """Entry point"""
    parser = ArgumentParser(description='ABIDE2BIDS downloader',
                            formatter_class=RawTextHelpFormatter)
    g_input = parser.add_argument_group('Inputs')
    g_input.add_argument('-i', '--input-abide-catalog', action='store',
                         required=True)
    g_input.add_argument('-n', '--dataset-name', action='store',
                         default='ABIDE Dataset')
    g_input.add_argument('-u', '--nitrc-user', action='store',
                         default=os.getenv('NITRC_USER'))
    g_input.add_argument('-p', '--nitrc-password', action='store',
                         default=os.getenv('NITRC_PASSWORD'))

    g_outputs = parser.add_argument_group('Outputs')
    g_outputs.add_argument('-o', '--output-dir', action='store',
                           default='ABIDE-BIDS')

    opts = parser.parse_args()

    if opts.nitrc_user is None or opts.nitrc_password is None:
        raise RuntimeError('NITRC user and password are required')

    dataset_desc = {'BIDSVersion': '1.0.0rc3',
                    'License': 'CC Attribution-NonCommercial-ShareAlike 3.0 Unported',
                    'Name': opts.dataset_name}

    out_dir = op.abspath(opts.output_dir)
    try:
        os.makedirs(out_dir)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise exc

    with open(op.join(out_dir, 'dataset_description.json'), 'w') as dfile:
        json.dump(dataset_desc, dfile)

    catalog = et.parse(opts.input_abide_catalog).getroot()
    urls = [el.get('URI') for el in catalog.iter() if el.get('URI') is not None]

    pool = Pool()
    args_list = [(url, opts.nitrc_user, opts.nitrc_password, out_dir)
                 for url in urls]
    res = pool.map(fetch, args_list)

    tsv_data = np.array([('subject_id', 'site_name')] + res)
    np.savetxt(op.join(out_dir, 'participants.tsv'), tsv_data, fmt='%s', delimiter='\t')


def fetch(args):
    """ Downloads a subject and formats it into BIDS """
    out_dir = None
    if len(args) == 3:
        url, user, password = args
    else:
        url, user, password, out_dir = args

    tmpdir = tempfile.mkdtemp()
    if out_dir is None:
        out_dir = os.getcwd()
    else:
        out_dir = op.abspath(out_dir)

    pkg_id = [u[9:] for u in url.split('/') if u.startswith('NITRC_IR_')][0]
    sub_file = op.join(tmpdir, '%s.zip' % pkg_id)

    cmd = ['curl', '-s', '-u', '%s:%s' % (user, password), '-o', sub_file, url]
    sp.check_call(cmd)
    sp.check_call(['unzip', '-qq', '-d', tmpdir, '-u', sub_file])

    abide_root = op.join(tmpdir, 'ABIDE')
    files = []
    for root, path, fname in os.walk(abide_root):
        if fname and (fname[0].endswith('nii') or fname[0].endswith('nii.gz')):
            if path:
                root = op.join(root, path[0])
            files.append(op.join(root, fname[0]))

    site_name, sub_str = files[0][len(abide_root) + 1:].split('/')[0].split('_')
    subject_id = 'sub-' + sub_str

    for i in files:
        ext = '.nii.gz'
        if i.endswith('.nii'):
            ext = '.nii'
        if 'mprage' in i:
            bids_dir = op.join(out_dir, subject_id, 'anat')
            try:
                os.makedirs(bids_dir)
            except OSError as exc:
                if exc.errno != errno.EEXIST:
                    raise exc
            shutil.copy(i, op.join(bids_dir, subject_id + '_T1w' + ext))

        if 'rest' in i:
            bids_dir = op.join(out_dir, subject_id, 'func')
            try:
                os.makedirs(bids_dir)
            except OSError as exc:
                if exc.errno != errno.EEXIST:
                    raise exc
            shutil.copy(i, op.join(bids_dir, subject_id + '_rest_bold' + ext))

    shutil.rmtree(tmpdir, ignore_errors=True, onerror=_myerror)

    print('Successfully processed subject %s from site %s' % (subject_id[4:], site_name))
    return subject_id[4:], site_name


def _myerror(msg):
    print('WARNING: Error deleting temporal files: %s' % msg)


if __name__ == '__main__':
    main()
