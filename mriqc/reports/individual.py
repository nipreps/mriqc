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

def individual_html(in_iqms, in_plots=None):
    import os.path as op  #pylint: disable=W0404
    import datetime
    from json import load
    from mriqc import __version__ as ver
    from mriqc.utils.misc import BIDS_COMP
    from mriqc.reports.utils import iqms2html, read_report_snippet
    from mriqc.data import IndividualTemplate
    from mriqc import logging
    from io import open  #pylint: disable=W0622
    report_log = logging.getLogger('mriqc.report')
    report_log.setLevel(logging.INFO)

    def _get_details(metadata, modality):
        warn_dict = metadata.pop('warnings', None)
        sett_dict = metadata.pop('settings', None)

        wf_details = []
        if modality == 'bold':
            exclude_index = sett_dict.pop('exclude_index')
            if exclude_index is None:
                report_log.warning('Building bold report: no exclude index was found')
            elif exclude_index > 0:
                msg = """\
<span class="problematic">Non-steady state (strong T1 contrast) has been detected in the \
first {} volumes</span>. They were excluded before generating any QC measures and plots."""
                wf_details.append(msg.format(exclude_index))

            hmc_fsl = sett_dict.pop('hmc_fsl')
            if hmc_fsl is not None:
                msg = 'Framewise Displacement was computed using '
                if hmc_fsl:
                    msg += 'FSL <code>mcflirt</code>'
                else:
                    msg += 'AFNI <code>3dvolreg</code>'
                wf_details.append(msg)

            fd_thres = sett_dict.pop('fd_thres')
            if fd_thres is not None:
                wf_details.append(
                    'Framewise Displacement threshold was defined at %f mm' % fd_thres)
        elif modality in ('T1w', 'T2w'):
            if warn_dict.pop('small_air_mask', False):
                wf_details.append(
                    '<span class="problematic">Detected hat mask was too small</span>')

            if warn_dict.pop('large_rot_frame', False):
                wf_details.append(
                    '<span class="problematic">Detected a zero-filled frame, has the original '
                    'image been rotated?</span>')

        return metadata, wf_details



    with open(in_iqms) as jsonfile:
        iqms_dict = load(jsonfile)

    # Now, the in_iqms file should be correctly named
    fname = op.splitext(op.basename(in_iqms))[0]
    out_file = op.abspath(fname + '.html')

    if in_plots is None:
        in_plots = []

    # Extract and prune metadata
    metadata = iqms_dict.pop('metadata', None)
    mod = metadata.pop('modality', None)
    metadata, wf_details = _get_details(metadata, mod)

    file_id = [metadata.pop(k, None)
               for k in list(BIDS_COMP.keys())]
    file_id = [comp for comp in file_id if comp is not None]

    pred_qa = None #metadata.pop('mriqc_pred', None)

    config = {
        'modality': mod,
        'sub_id': '_'.join(file_id),
        'timestamp': datetime.datetime.now().strftime("%Y-%m-%d, %H:%M"),
        'version': ver,
        'imparams': iqms2html(iqms_dict, 'iqms-table'),
        'svg_files': [read_report_snippet(pfile) for pfile in in_plots],
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
