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

def individual_html(in_iqms, in_metadata=None, in_plots=None, exclude_index=0,
                    wf_details=None):
    import os.path as op  #pylint: disable=W0404
    import datetime
    import re
    from json import load
    from mriqc import __version__ as ver
    from mriqc.reports.utils import iqms2html
    from mriqc.data import IndividualTemplate
    from mriqc import logging
    from io import open  #pylint: disable=W0622
    MRIQC_REPORT_LOG = logging.getLogger('mriqc.report')
    MRIQC_REPORT_LOG.setLevel(logging.INFO)

    with open(in_iqms) as f:
        iqms_dict = load(f)

    if in_plots is None:
        in_plots = []

    if wf_details is None:
        wf_details = []

    svg_files = []
    for pfile in in_plots:
        with open(pfile) as f:
            svg_content_lines = f.read().split('\n')
            svg_lines_corrected = []
            for line in svg_content_lines:
                if "<svg " in line:
                    line = re.sub(' height="[0-9.]+[a-z]*"', '', line)
                    line = re.sub(' width="[0-9.]+[a-z]*"', '', line)
                svg_lines_corrected.append(line)

            svg_files.append('\n'.join(svg_lines_corrected))

    name_els = [iqms_dict.pop(k, None) for k in [
        'qc_type', 'subject_id', 'session_id', 'task_id', 'run_id']]
    if not name_els[1].startswith('sub-'):
        name_els[1] = 'sub-' + name_els[1]

    MRIQC_REPORT_LOG.info('Elements %s', [el for el in name_els if el is not None])

    if name_els[0].startswith('anat'):
        name_els[0] = 'anatomical'
        msk_vals = []
        for k in ['snr_d_csf', 'snr_d_gm', 'snr_d_wm', 'fber']:
            elements = k.split('_')
            iqm = iqms_dict[elements[0]]
            if len(elements) == 1:
                msk_vals.append(iqm < 0.)
            else:
                msk_vals.append(iqm['_'.join(elements[1:])] < 0.)

        if any(msk_vals):
            wf_details.append('Noise variance in the background is very low')
            if all(msk_vals):
                wf_details[-1] += (' for all measures: <span class="problematic">'
                                   'the original file could be masked</span>.')
            else:
                wf_details[-1] += '.'
    elif name_els[0].startswith('func'):
        name_els[0] = 'functional'
    else:
        RuntimeError('Unknown QC type "%s"' % name_els[0])


    out_file = op.abspath('_'.join([el for el in name_els if el is not None]) + '_report.html')

    tpl = IndividualTemplate()
    tpl.generate_conf({
            'qctype': name_els[0],
            'sub_id': name_els[1][4:],
            'timestamp': datetime.datetime.now().strftime("%Y-%m-%d, %H:%M"),
            'version': ver,
            'imparams': iqms2html(iqms_dict, 'iqms-table'),
            'svg_files': svg_files,
            'exclude_index': exclude_index,
            'workflow_details': wf_details,
            'metadata': iqms2html(in_metadata, 'metadata-table'),
        }, out_file)

    MRIQC_REPORT_LOG.info('Generated individual log (%s)', out_file)
    return out_file
