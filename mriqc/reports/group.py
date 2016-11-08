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

from sys import version_info
from builtins import zip, object, str, bytes  # pylint: disable=W0622

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
    from mriqc import __version__ as ver
    from mriqc.data import GroupTemplate
    from mriqc.utils.misc import check_folder

    if version_info[0] > 2:
        from io import StringIO as TextIO
    else:
        from io import BytesIO as TextIO

    QCGROUPS = {
        'anat': [
            (['cjv'], None),
            (['cnr'], None),
            (['efc'], None),
            (['fber'], None),
            (['wm2max'], None),
            (['snr_csf', 'snr_gm', 'snr_wm'], None),
            (['snr_d_csf', 'snr_d_gm', 'snr_d_wm'], None),
            (['fwhm_avg', 'fwhm_x', 'fwhm_y', 'fwhm_z'], 'mm'),
            (['qi1', 'qi2'], None),
            (['inu_range', 'inu_med'], None),
            (['icvs_csf', 'icvs_gm', 'icvs_wm'], None),
            (['rpve_csf', 'rpve_gm', 'rpve_wm'], None),
            (['tpm_overlap_csf', 'tpm_overlap_gm', 'tpm_overlap_wm'], None),
            (['summary_bg_mean', 'summary_bg_stdv', 'summary_bg_k',
              'summary_bg_p05', 'summary_bg_p95'], None),
            (['summary_csf_mean', 'summary_csf_stdv', 'summary_csf_k',
              'summary_csf_p05', 'summary_csf_p95'], None),
            (['summary_gm_mean', 'summary_gm_stdv', 'summary_gm_k',
              'summary_gm_p05', 'summary_gm_p95'], None),
            (['summary_wm_mean', 'summary_wm_stdv', 'summary_wm_k',
              'summary_wm_p05', 'summary_wm_p95'], None)
        ],
        'func': [
            (['efc'], None),
            (['fber'], None),
            (['fwhm', 'fwhm_x', 'fwhm_y', 'fwhm_z'], 'mm'),
            (['gsr_%s' % a for a in ['x', 'y']], None),
            (['snr'], None),
            (['dvars_std', 'dvars_vstd'], None),
            (['dvars_nstd'], None),
            (['fd_mean'], 'mm'),
            (['fd_num'], '# timepoints'),
            (['fd_perc'], '% timepoints'),
            (['spikes_num'], '# slices'),
            (['gcor'], None),
            (['tsnr'], None),
            (['aor'], None),
            (['aqi'], None),
            (['summary_bg_mean', 'summary_bg_stdv', 'summary_bg_k',
              'summary_bg_p05', 'summary_bg_p95'], None),
            (['summary_fg_mean', 'summary_fg_stdv', 'summary_fg_k',
              'summary_fg_p05', 'summary_fg_p95'], None),
        ]
    }

    dataframe = pd.read_csv(csv_file, index_col=False)

    # format participant labels
    formatter = lambda row: '{subject_id}_ses-{session_id}_run-{run_id}'.format(**row)
    dataframe['label'] = dataframe[['subject_id', 'session_id', 'run_id']].apply(formatter, axis=1)
    nPart = len(dataframe)

    csv_groups = []
    for group, units in QCGROUPS[qctype]:
        dfdict = {'iqm': [], 'value': [], 'label': [], 'units': []}

        for iqm in group:
            if iqm in dataframe.columns.ravel().tolist():
                values = dataframe[[iqm]].values.ravel().tolist()
                dfdict['iqm'] += [iqm] * nPart
                dfdict['units'] += [units] * nPart
                dfdict['value'] += values
                dfdict['label'] += dataframe[['label']].values.ravel().tolist()

        csv_df = pd.DataFrame(dfdict)
        csv_str = TextIO()
        csv_df[['iqm', 'value', 'label', 'units']].to_csv(csv_str, index=False)
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
