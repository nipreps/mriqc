#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
# pylint: disable=no-member
#
# @Author: oesteban
# @Date:   2016-01-05 11:33:39
# @Email:  code@oscaresteban.es
# @Last modified by:   oesteban
""" Encapsulates report generation functions """
from __future__ import print_function, division, absolute_import, unicode_literals
import os
from sys import version_info
import os.path as op
from builtins import zip, range, object, str, bytes  # pylint: disable=W0622

from mriqc import logging
MRIQC_REPORT_LOG = logging.getLogger('mriqc.report')
MRIQC_REPORT_LOG.setLevel(logging.INFO)

def gen_html(csv_file, qctype, out_file=None):
    import os.path as op
    from os import remove
    from shutil import copy, rmtree
    import datetime
    from pkg_resources import resource_filename as pkgrf
    import pandas as pd
    import numpy as np
    from mriqc import __version__ as ver
    from mriqc.data import GroupTemplate
    from mriqc.utils.misc import check_folder

    if version_info[0] > 2:
        from io import StringIO as TextIO
    else:
        from io import BytesIO as TextIO

    QCGROUPS = {
        'anat': [
            ['cjv'],
            ['cnr'],
            ['efc'],
            ['fber'],
            ['wm2max'],
            ['snr_csf', 'snr_gm', 'snr_wm'],
            ['fwhm_avg', 'fwhm_x', 'fwhm_y', 'fwhm_z'],
            ['qi1', 'qi2'],
            ['inu_range', 'inu_med'],
            ['icvs_csf', 'icvs_gm', 'icvs_wm'],
            ['rpve_csf', 'rpve_gm', 'rpve_wm'],
            ['summary_mean_bg', 'summary_stdv_bg', 'summary_k_bg', 'summary_p05_bg', 'summary_p95_bg'],
            ['summary_mean_csf', 'summary_stdv_csf', 'summary_k_csf',
             'summary_p05_csf', 'summary_p95_csf'],
            ['summary_mean_gm', 'summary_stdv_gm', 'summary_k_gm', 'summary_p05_gm', 'summary_p95_gm'],
            ['summary_mean_wm', 'summary_stdv_wm', 'summary_k_wm', 'summary_p05_wm', 'summary_p95_wm']
        ],
        'func': [
            ['efc'],
            ['fber'],
            ['fwhm', 'fwhm_x', 'fwhm_y', 'fwhm_z'],
            ['gsr_%s' % a for a in ['x', 'y']],
            ['snr'],
            ['dvars_std', 'dvars_vstd'],
            ['dvars_nstd'],
            ['fd_mean'],
            ['fd_num'],
            ['fd_perc'],
            ['gcor'],
            ['m_tsnr'],
            ['outlier'],
            ['quality'],
            ['summary_mean_bg', 'summary_stdv_bg', 'summary_k_bg',
             'summary_p05_bg', 'summary_p95_bg'],
            ['summary_mean_fg', 'summary_stdv_fg', 'summary_k_fg',
             'summary_p05_fg', 'summary_p95_fg'],
        ]
    }

    dataframe = pd.read_csv(csv_file, index_col=False)

    # format participant labels
    formatter = lambda row: '{subject_id}_ses-{session_id}_run-{run_id}'.format(**row)
    dataframe['label'] = dataframe[['subject_id', 'session_id', 'run_id']].apply(formatter, axis=1)
    nPart = len(dataframe)

    csv_groups = []
    for group in QCGROUPS[qctype]:
        dfdict = {'iqm': [], 'value': [], 'label': []}

        for iqm in group:
            if iqm in dataframe.columns.ravel().tolist():
                values = dataframe[[iqm]].values.ravel().tolist()
                dfdict['iqm'] += [iqm] * nPart
                dfdict['value'] += values
                dfdict['label'] += dataframe[['label']].values.ravel().tolist()

        csv_df = pd.DataFrame(dfdict)
        csv_str = TextIO()
        csv_df.to_csv(csv_str, index=False)
        csv_groups.append(csv_str.getvalue())

    if out_file is None:
        out_file = op.abspath('group.html')
    tpl = GroupTemplate()
    tpl.generate_conf({
            'qctype': 'anatomical' if qctype == 'anat' else 'functional',
            'timestamp': datetime.datetime.now().strftime("%Y-%m-%d, %H:%M"),
            'version': ver,
            'csv_groups': csv_groups,

        }, out_file)

    res_folder = op.join(op.dirname(out_file), 'resources')
    check_folder(res_folder)
    for fname in ['boxplots.css', 'boxplots.js', 'd3.min.js']:
        dstpath = op.join(res_folder, fname)
        if op.isfile(dstpath):
            remove(dstpath)

        copy(pkgrf('mriqc', op.join('data', 'reports', 'resources', fname)), dstpath)

    MRIQC_REPORT_LOG.info('Generated group-level report (%s)', out_file)
    return out_file
