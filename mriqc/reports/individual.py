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

def individual_html(in_iqms, in_plots=None, exclude_index=0, wf_details=None):
    import os.path as op  #pylint: disable=W0404
    import datetime
    from json import load
    from mriqc import __version__ as ver
    from mriqc.utils.misc import BIDS_COMP
    from mriqc.reports.utils import iqms2html, anat_flags, read_report_snippet
    from mriqc.data import IndividualTemplate
    from mriqc import logging
    from io import open  #pylint: disable=W0622
    report_log = logging.getLogger('mriqc.report')
    report_log.setLevel(logging.INFO)

    with open(in_iqms) as jsonfile:
        iqms_dict = load(jsonfile)

    # Now, the in_iqms file should be correctly named
    fname = op.splitext(op.basename(in_iqms))[0]
    out_file = op.abspath(fname + '.html')

    if in_plots is None:
        in_plots = []

    if wf_details is None:
        wf_details = []

    # Extract and prune metadata
    metadata = iqms_dict.pop('metadata', None)
    mod = metadata.pop('modality', None)
    file_id = [metadata.pop(k, None)
               for k in list(BIDS_COMP.keys())]
    file_id = [comp for comp in file_id if comp is not None]

    pred_qa = metadata.pop('mriqc_pred', None)

    # Deal with special IQMs
    if mod in ('T1w', 'T2w'):
        flags = anat_flags(iqms_dict)
        if flags:
            wf_details.append(flags)
    elif mod == 'bold':
        pass
    else:
        RuntimeError('Unknown modality "%s"' % mod)

    config = {
        'modality': mod,
        'sub_id': '_'.join(file_id),
        'timestamp': datetime.datetime.now().strftime("%Y-%m-%d, %H:%M"),
        'version': ver,
        'imparams': iqms2html(iqms_dict, 'iqms-table'),
        'svg_files': [read_report_snippet(pfile) for pfile in in_plots],
        'exclude_index': exclude_index,
        'workflow_details': wf_details,
        'metadata': iqms2html(metadata, 'metadata-table'),
        'pred_qa': pred_qa
    }

    if config['metadata'] is None:
        config['workflow_details'].append(
            '<span class="warning">File has no metadata</span> '
            '<span>(sidecar JSON file missing or empty)</span>')

    tpl = IndividualTemplate()
    tpl.generate_conf(config, out_file)

    report_log.info('Generated individual log (%s)', out_file)
    return out_file
