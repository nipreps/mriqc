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
# @Last Modified time: 2016-04-13 15:34:59
""" Encapsulates report generation functions """

import sys
import os
import os.path as op
import pandas as pd
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from .utils import read_csv, find_failed, image_parameters
from ..interfaces.viz_utils import plot_measures, plot_all

# matplotlib.rc('figure', figsize=(11.69, 8.27))  # for DINA4 size
STRUCTURAL_QCGROUPS = [
    ['icvs_csf', 'icvs_gm', 'icvs_wm'],
    ['rpve_csf', 'rpve_gm', 'rpve_wm'],
    ['inu_range', 'inu_med'],
    ['cnr'], ['efc'], ['fber'], ['cjv'],
    ['fwhm', 'fwhm_x', 'fwhm_y', 'fwhm_z'],
    ['qi1', 'qi2'],
    ['snr', 'snr_csf', 'snr_gm', 'snr_wm'],
    ['summary_mean_bg', 'summary_stdv_bg', 'summary_p05_bg', 'summary_p95_bg'],
    ['summary_mean_csf', 'summary_stdv_csf', 'summary_p05_csf', 'summary_p95_csf'],
    ['summary_mean_gm', 'summary_stdv_gm', 'summary_p05_gm', 'summary_p95_gm'],
    ['summary_mean_wm', 'summary_stdv_wm', 'summary_p05_wm', 'summary_p95_wm']
]

FUNC_SPATIAL_QCGROUPS = [
    ['summary_mean_bg', 'summary_stdv_bg', 'summary_p05_bg', 'summary_p95_bg'],
    ['summary_mean_fg', 'summary_stdv_fg', 'summary_p05_fg', 'summary_p95_fg'],
    ['efc'],
    ['fber'],
    ['fwhm', 'fwhm_x', 'fwhm_y', 'fwhm_z'],
    ['gsr_%s' % a for a in ['x', 'y']],
    ['snr']
]

FUNC_TEMPORAL_QCGROUPS = [
    ['dvars'], ['gcor'], ['m_tsnr'], ['mean_fd'],
    ['num_fd'], ['outlier'], ['perc_fd'], ['quality']
]

def workflow_report(in_csv, qctype, sub_list=None, settings=None):
    """ Creates the report """
    import datetime

    if settings is None:
        settings = {}

    out_dir = settings.get('output_dir', os.getcwd())
    work_dir = settings.get('work_dir', op.abspath('tmp'))
    out_file = op.join(out_dir, qctype + '_%s.pdf')
    dframe, subject_list = read_csv(in_csv)

    result = {}
    func = getattr(sys.modules[__name__], 'report_' + qctype)

    failed_str = 'none'
    if sub_list is not None:
        failed = find_failed(dframe, sub_list)
        if len(failed) > 0:
            failed_str = ', '.join(sorted(['%s_%s_%s' % f for f in failed]))


    imparams = image_parameters(dframe)
    pdf_group = []
    # Generate summary page
    out_sum = op.join(work_dir, 'summary_group.pdf')
    summary_cover({'modality': qctype, 'failed': failed_str, 'params': imparams},
                  is_group=True, out_file=out_sum)
    pdf_group.append(out_sum)

    # Generate group report
    qc_group = op.join(work_dir, 'qc_measures_group.pdf')
    # Generate violinplots. If successfull, add documentation.
    func(dframe, out_file=qc_group)
    pdf_group.append(qc_group)

    if len(pdf_group) > 0:
        out_group_file = op.join(out_dir, '%s_group.pdf' % qctype)
        # Generate final report with collected pdfs in plots
        concat_pdf(pdf_group, out_group_file)
        result['group'] = {'success': True, 'path': out_group_file}

    out_indiv_files = []
    # Generate individual reports for subjects
    for subid in subject_list:
        # Get subject-specific info
        subdf = dframe.loc[dframe['subject'] == subid]
        sessions = sorted(pd.unique(subdf.session.ravel()))
        plots = []
        sess_scans = []
        subparams = {}
        # Re-build mosaic location
        for sesid in sessions:
            sesdf = subdf.loc[subdf['session'] == sesid]
            scans = sorted(pd.unique(sesdf.scan.ravel()))

            # Each scan has a volume and (optional) fd plot
            for scanid in scans:
                subparams[(sesid, scanid)] = imparams[(subid, sesid, scanid)]
                if 'anat' in qctype:
                    fpdf = op.join(work_dir, 'anatomical_%s_%s_%s.pdf' %
                                   (subid, sesid, scanid))

                    if op.isfile(fpdf):
                        plots.append(fpdf)

                if 'func' in qctype:
                    mepi = op.join(work_dir, 'meanepi_%s_%s_%s.pdf' %
                                   (subid, sesid, scanid))
                    if op.isfile(mepi):
                        plots.append(mepi)

                    tsnr = op.join(work_dir, 'tsnr_%s_%s_%s.pdf' %
                                   (subid, sesid, scanid))
                    if op.isfile(tsnr):
                        plots.append(tsnr)

                    framedisp = op.join(work_dir, 'fd_%s_%s_%s.pdf' %
                                        (subid, sesid, scanid))
                    if op.isfile(framedisp):
                        plots.append(framedisp)

            sess_scans.append('%s (%s)' % (sesid, ', '.join(scans)))

        # Summary cover
        sfailed = []
        if len(failed) > 0:
            sfailed = ['%s (%s)' % (s[1], s[2])
                       for s in failed if subid == s[0]]
        out_sum = op.join(work_dir, '%s_summary_%s.pdf' % (qctype, subid))
        summary_cover(
            {'sub_id': subid, 'modality': qctype, 'included': ", ".join(sess_scans),
             'failed': ",".join(sfailed) if sfailed else "none",
             'params': subparams},
            out_file=out_sum)
        plots.insert(0, out_sum)

        # Summary (violinplots) of QC measures
        qc_ms = op.join(work_dir, '%s_measures_%s.pdf' % (qctype, subid))

        func(dframe, subject=subid, out_file=qc_ms)
        plots.append(qc_ms)

        if len(plots) > 0:
            # Generate final report with collected pdfs in plots
            sub_path = out_file % subid
            concat_pdf(plots, sub_path)
            out_indiv_files.append(sub_path)
            result[subid] = {'success': True, 'path': sub_path}
    return out_group_file, out_indiv_files, result


def summary_cover(data, is_group=False, out_file=None):
    """ Generates a cover page with subject information """
    import datetime
    import codecs
    import StringIO
    from xhtml2pdf import pisa  # pylint: disable=no-name-in-module

    # open output file for writing (truncated binary)
    result = open(out_file, "w+b")

    substr = '<table><tr>'
    if is_group:
        substr += '<th>Subject ID</th>'
    substr += ('<th>Session</th><th>Scan ID</th><th>Image size (voxels)</th><th>Spacing (mm)</th>'
               '<th>TR (ms)</th><th>Time steps</th></tr>')


    for k in sorted(data['params'].keys()):
        info = data['params'][k]

        if is_group:
            substr += '<tr><td>%s</td><td>%s</td><td>%s</td>' % k
        else:
            substr += '<tr><td>%s</td><td>%s</td>' % k
        substr += '<td>{size:s}</td><td>{spacing:s}</td>'.format(**info)
        substr += '<td>%f</td>' % info['tr'] if 'tr' in info.keys() else '<td>N/A</td>'
        substr += '<td>%d</td>' % info['size_t'] if 'size_t' in info.keys() else '<td>1</td>'
        substr += '</tr>\n'
    substr += '</table>'

    html_dir = op.abspath(
        op.join(op.dirname(__file__), 'html', 'cover_group.html'
                if is_group else 'cover_subj.html'))

    with codecs.open(html_dir, mode='r', encoding='utf-8') as ftpl:
        html = ftpl.read().format

    if is_group:
        values = {'imparams': substr, 'modality': data['modality'], 'failed': data['failed'],
                  'timestamp': datetime.datetime.now().strftime("%Y-%m-%d, %H:%M")}
    else:
        values = {'sub_id': data['sub_id'], 'imparams': substr, 'modality': data['modality'],
                  'timestamp': datetime.datetime.now().strftime("%Y-%m-%d, %H:%M"),
                  'failed': data['failed']}

    # convert HTML to PDF
    status = pisa.pisaDocument(html(**values), result, encoding='UTF-8')
    result.close()

    # return True on success and False on errors
    return status.err


def concat_pdf(in_files, out_file='concatenated.pdf'):
    """ Concatenate PDF list (http://stackoverflow.com/a/3444735) """
    from PyPDF2 import PdfFileWriter, PdfFileReader
    outpdf = PdfFileWriter()

    for in_file in in_files:
        inpdf = PdfFileReader(file(in_file, 'rb'))
        for fpdf in range(inpdf.numPages):
            outpdf.addPage(inpdf.getPage(fpdf))
    outpdf.write(file(out_file, 'wb'))
    return out_file


def _write_report(dframe, groups, sub_id=None, sc_split=False, condensed=True,
                  out_file='report.pdf'):
    """ Generates the violin plots of each qctype """
    columns = dframe.columns.ravel()
    headers = []
    for group in groups:
        rem = []
        for head in group:
            if head not in columns:
                rem.append(head)
            else:
                headers.append(head)
        for i in rem:
            group.remove(i)

    report = PdfPages(out_file)
    sessions = sorted(pd.unique(dframe.session.ravel()))
    for ssid in sessions:
        sesdf = dframe.copy().loc[dframe['session'] == ssid]
        scans = pd.unique(sesdf.scan.ravel())
        if sc_split:
            for scid in scans:
                subset = sesdf.loc[sesdf['scan'] == scid]
                if len(subset.index) > 1:
                    if sub_id is None:
                        subtitle = '(%s_%s)' % (ssid, scid)
                    else:
                        subtitle = '(subject %s_%s_%s)' % (sub_id, ssid, scid)
                    if condensed:
                        fig = plot_all(sesdf, groups, subject=sub_id,
                                       title='QC measures ' + subtitle)
                    else:
                        fig = plot_measures(
                            sesdf, headers, subject=sub_id,
                            title='QC measures ' + subtitle)
                    report.savefig(fig, dpi=300)
                    fig.clf()
        else:
            if len(sesdf.index) > 1:
                if sub_id is None:
                    subtitle = '(%s)' % (ssid)
                else:
                    subtitle = '(subject %s_%s)' % (sub_id, ssid)
                if condensed:
                    fig = plot_all(sesdf, groups, subject=sub_id,
                                   title='QC measures ' + subtitle)
                else:
                    fig = plot_measures(
                        sesdf, headers, subject=sub_id,
                        title='QC measures ' + subtitle)
                report.savefig(fig, dpi=300)
                fig.clf()

    report.close()
    plt.close()
    # print 'Written report file %s' % out_file
    return out_file

def report_anatomical(
        dframe, subject=None, sc_split=False, condensed=True,
        out_file='anatomical.pdf'):
    """ Calls the report generator on the functional measures """
    return _write_report(dframe, STRUCTURAL_QCGROUPS, sub_id=subject, sc_split=sc_split,
                         condensed=condensed, out_file=out_file)


def report_functional(
        dframe, subject=None, sc_split=False, condensed=True,
        out_file='functional.pdf'):
    """ Calls the report generator on the functional measures """
    from tempfile import mkdtemp

    wdir = mkdtemp()
    fspatial = _write_report(
        dframe, FUNC_TEMPORAL_QCGROUPS, sub_id=subject, sc_split=sc_split,
        condensed=condensed, out_file=op.join(wdir, 'fspatial.pdf'))

    ftemporal = _write_report(
        dframe, FUNC_SPATIAL_QCGROUPS, sub_id=subject, sc_split=sc_split,
        condensed=condensed, out_file=op.join(wdir, 'ftemporal.pdf'))

    concat_pdf([fspatial, ftemporal], out_file)
    return out_file
