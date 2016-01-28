#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# @Author: oesteban
# @Date:   2016-01-05 11:33:39
# @Email:  code@oscaresteban.es
# @Last modified by:   oesteban
# @Last Modified time: 2016-01-18 18:05:18


import sys
import os
import os.path as op
import pandas as pd
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from .interfaces.viz_utils import (plot_measures, plot_mosaic, plot_all,
                                   plot_fd, plot_dist)

# matplotlib.rc('figure', figsize=(11.69, 8.27))  # for DINA4 size


def workflow_report(in_csv, qap_type, sub_list=[], settings={}):
    import datetime
    import numpy as np

    out_dir = settings.get('output_dir', os.getcwd())
    work_dir = settings.get('work_dir', op.abspath('tmp'))
    out_file = op.join(out_dir, qap_type + '_%s.pdf')

    # Read csv file, sort and drop duplicates
    try:
        df = pd.read_csv(in_csv, dtype={'subject': str}).sort_values(
            by=['subject', 'session', 'scan'])
    except AttributeError:
        df = pd.read_csv(in_csv, dtype={'subject': str}).sort(
            columns=['subject', 'session', 'scan'])

    try:
        df.drop_duplicates(['subject', 'session', 'scan'], keep='last',
                           inplace=True)
    except TypeError:
        df.drop_duplicates(['subject', 'session', 'scan'], take_last=True,
                           inplace=True)

    subject_list = sorted(pd.unique(df.subject.ravel()))
    result = {}
    func = getattr(sys.modules[__name__], 'report_' + qap_type)

    # Identify failed subjects
    failed_str = "none"
    if sub_list:
        # Remove undesired elements from sub_list tuples
        sub_list = [(s[0], s[1], s[2]) for s in sub_list]
        success = [tuple(x) for x in df[['subject', 'session', 'scan']].values]
        failed = set(sub_list) - set(success)

        if len(failed) > 0:
            failed_str = ', '.join(sorted(['%s_%s_%s' % f for f in failed]))

    pdf_group = []

    # Generate summary page
    out_sum = op.join(work_dir, 'summary_group.pdf')
    summary_cover(
        (qap_type, datetime.datetime.now().strftime("%Y-%m-%d, %H:%M"),
         failed_str), is_group=True, out_file=out_sum)
    pdf_group.append(out_sum)

    # Generate group report
    qc_group = op.join(work_dir, 'qc_measures_group.pdf')
    # Generate violinplots. If successfull, add documentation.
    func(df, out_file=qc_group)
    pdf_group.append(qc_group)

    # Generate documentation page
    doc = op.join(work_dir, '%s_doc.pdf' % qap_type)

    # Let documentation page fail
    get_documentation(qap_type, doc)
    if doc is not None:
        pdf_group.append(doc)

    if len(pdf_group) > 0:
        out_group_file = op.join(out_dir, '%s_group.pdf' % qap_type)
        # Generate final report with collected pdfs in plots
        concat_pdf(pdf_group, out_group_file)
        result['group'] = {'success': True, 'path': out_group_file}

    # Generate individual reports for subjects
    for subid in subject_list:
        # Get subject-specific info
        subdf = df.loc[df['subject'] == subid]
        sessions = sorted(pd.unique(subdf.session.ravel()))
        plots = []
        sess_scans = []
        # Re-build mosaic location
        for sesid in sessions:
            sesdf = subdf.loc[subdf['session'] == sesid]
            scans = sorted(pd.unique(sesdf.scan.ravel()))

            # Each scan has a volume and (optional) fd plot
            for scanid in scans:
                if 'anat' in qap_type:
                    m = op.join(work_dir, 'anatomical_%s_%s_%s.pdf' %
                                (subid, sesid, scanid))

                    if op.isfile(m):
                        plots.append(m)

                if 'func' in qap_type:
                    mepi = op.join(work_dir, 'meanepi_%s_%s_%s.pdf' %
                                   (subid, sesid, scanid))
                    if op.isfile(mepi):
                        plots.append(mepi)

                    tsnr = op.join(work_dir, 'tsnr_%s_%s_%s.pdf' %
                                   (subid, sesid, scanid))
                    if op.isfile(tsnr):
                        plots.append(tsnr)

                    fd = op.join(work_dir, 'fd_%s_%s_%s.pdf' %
                                 (subid, sesid, scanid))
                    if op.isfile(fd):
                        plots.append(fd)

            sess_scans.append('%s (%s)' % (sesid, ', '.join(scans)))

        # Summary cover
        sfailed = []
        if len(failed) > 0:
            sfailed = ['%s (%s)' % (s[1], s[2])
                       for s in failed if subid == s[0]]
        out_sum = op.join(work_dir, '%s_summary_%s.pdf' % (qap_type, subid))
        summary_cover(
            (subid, subid, qap_type,
             datetime.datetime.now().strftime("%Y-%m-%d, %H:%M"),
             ", ".join(sess_scans),
             ",".join(sfailed) if sfailed else "none"),
            out_file=out_sum)
        plots.insert(0, out_sum)

        # Summary (violinplots) of QC measures
        qc_ms = op.join(work_dir, '%s_measures_%s.pdf' % (qap_type, subid))

        func(df, subject=subid, out_file=qc_ms)
        plots.append(qc_ms)

        if len(plots) > 0:
            if doc is not None:
                plots.append(doc)

            # Generate final report with collected pdfs in plots
            sub_path = out_file % subid
            concat_pdf(plots, sub_path)
            result[subid] = {'success': True, 'path': sub_path}
    return result


def get_documentation(doc_type, out_file):
    import codecs
    import StringIO
    from xhtml2pdf import pisa
    # open output file for writing (truncated binary)
    result = open(out_file, "w+b")

    html_dir = op.abspath(
        op.join(op.dirname(__file__), 'html', '%s.html' % doc_type))

    with codecs.open(html_dir, mode='r', encoding='utf-8') as f:
        html = f.read()

    # convert HTML to PDF
    status = pisa.pisaDocument(html, result, encoding='UTF-8')
    result.close()

    # return True on success and False on errors
    return status.err


def summary_cover(data, is_group=False, out_file=None):
    import codecs
    import StringIO
    from xhtml2pdf import pisa
    # open output file for writing (truncated binary)
    result = open(out_file, "w+b")

    html_file = 'cover_group.html' if is_group else 'cover_subj.html'

    html_dir = op.abspath(
        op.join(op.dirname(__file__), 'html', html_file))

    with codecs.open(html_dir, mode='r', encoding='utf-8') as f:
        html = f.read()

    # convert HTML to PDF
    status = pisa.pisaDocument(html % data, result, encoding='UTF-8')
    result.close()

    # return True on success and False on errors
    return status.err


def concat_pdf(in_files, out_file='concatenated.pdf'):
    """
    Concatenate PDF list (http://stackoverflow.com/a/3444735)
    """
    from PyPDF2 import PdfFileWriter, PdfFileReader
    outpdf = PdfFileWriter()

    for in_file in in_files:
        inpdf = PdfFileReader(file(in_file, 'rb'))
        for p in range(inpdf.numPages):
            outpdf.addPage(inpdf.getPage(p))
    outpdf.write(file(out_file, 'wb'))
    return out_file


def _write_report(df, groups, sub_id=None, sc_split=False, condensed=True,
                  out_file='report.pdf'):
    columns = df.columns.ravel()
    headers = []
    for g in groups:
        rem = []
        for h in g:
            if h not in columns:
                rem.append(h)
            else:
                headers.append(h)
        for r in rem:
            g.remove(r)

    report = PdfPages(out_file)
    sessions = sorted(pd.unique(df.session.ravel()))
    for ss in sessions:
        sesdf = df.copy().loc[df['session'] == ss]
        scans = pd.unique(sesdf.scan.ravel())
        if sc_split:
            for sc in scans:
                subset = sesdf.loc[sesdf['scan'] == sc]
                if len(subset.index) > 1:
                    if sub_id is None:
                        subtitle = '(%s_%s)' % (ss, sc)
                    else:
                        subtitle = '(subject %s_%s_%s)' % (sub_id, ss, sc)
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
                    subtitle = '(%s)' % (ss)
                else:
                    subtitle = '(subject %s_%s)' % (sub_id, ss)
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


def _write_all_reports(df, groups, sc_split=False, condensed=True,
                       out_file='report.pdf'):

    outlist = []
    _write_report(
        df, groups, sc_split=sc_split, condensed=condensed, out_file=out_file)

    subject_list = sorted(pd.unique(df.subject.ravel()))
    for sub_id in subject_list:
        tpl, _ = op.splitext(op.basename(out_file))
        tpl = op.join(op.dirname(out_file), tpl) + '_%s.pdf'
        outlist.append(_write_report(
            df, groups, sub_id=sub_id, sc_split=sc_split, condensed=condensed,
            out_file=tpl % sub_id))
    return out_file, outlist


def all_anatomical(df, sc_split=False, condensed=True,
                   out_file='anatomical.pdf'):
    groups = [['bg_size', 'fg_size'],
              ['bg_mean', 'fg_mean'],
              ['bg_std', 'fg_std'],
              ['csf_size', 'gm_size', 'wm_size'],
              ['csf_mean', 'gm_mean', 'wm_mean'],
              ['csf_std', 'gm_std', 'wm_std'],
              ['cnr'],
              ['efc'],
              ['fber'],
              ['fwhm', 'fwhm_x', 'fwhm_y', 'fwhm_z'],
              ['qi1'],
              ['snr']]
    return _write_all_reports(
        df, groups, sc_split=sc_split,
        condensed=condensed, out_file=out_file)


def all_func_temporal(df, sc_split=False, condensed=True,
                      out_file='func_temporal.pdf'):
    groups = [['dvars'], ['gcor'], ['m_tsnr'], ['mean_fd'],
              ['num_fd'], ['outlier'], ['perc_fd'], ['quality']]
    return _write_all_reports(
        df, groups, sc_split=sc_split,
        condensed=condensed, out_file=out_file)


def all_func_spatial(df, sc_split=False, condensed=False,
                     out_file='func_spatial.pdf'):
    groups = [['bg_size', 'fg_size'],
              ['bg_mean', 'fg_mean'],
              ['bg_std', 'fg_std'],
              ['efc'],
              ['fber'],
              ['fwhm', 'fwhm_x', 'fwhm_y', 'fwhm_z'],
              ['ghost_%s' % a for a in ['x', 'y', 'z']],
              ['snr']]
    return _write_all_reports(
        df, groups, sc_split=sc_split,
        condensed=condensed, out_file=out_file)


def report_anatomical(
        df, subject=None, sc_split=False, condensed=True,
        out_file='anatomical.pdf'):
    groups = [['bg_size', 'fg_size'],
              ['bg_mean', 'fg_mean'],
              ['bg_std', 'fg_std'],
              ['csf_size', 'gm_size', 'wm_size'],
              ['csf_mean', 'gm_mean', 'wm_mean'],
              ['csf_std', 'gm_std', 'wm_std'],
              ['cnr'],
              ['efc'],
              ['fber'],
              ['fwhm', 'fwhm_x', 'fwhm_y', 'fwhm_z'],
              ['qi1'],
              ['snr']]
    return _write_report(
        df, groups, sub_id=subject, sc_split=sc_split, condensed=condensed,
        out_file=out_file)


def report_functional(
        df, subject=None, sc_split=False, condensed=True,
        out_file='functional.pdf'):
    from tempfile import mkdtemp

    wd = mkdtemp()
    groups = [['dvars'], ['gcor'], ['m_tsnr'], ['mean_fd'],
              ['num_fd'], ['outlier'], ['perc_fd'], ['quality']]
    fspatial = _write_report(
        df, groups, sub_id=subject, sc_split=sc_split, condensed=condensed,
        out_file=op.join(wd, 'fspatial.pdf'))

    groups = [['bg_size', 'fg_size'],
              ['bg_mean', 'fg_mean'],
              ['bg_std', 'fg_std'],
              ['efc'],
              ['fber'],
              ['fwhm', 'fwhm_x', 'fwhm_y', 'fwhm_z'],
              ['ghost_%s' % a for a in ['x', 'y', 'z']],
              ['snr']]
    ftemporal = _write_report(
        df, groups, sub_id=subject, sc_split=sc_split, condensed=condensed,
        out_file=op.join(wd, 'ftemporal.pdf'))

    concat_pdf([fspatial, ftemporal], out_file)
    return out_file
