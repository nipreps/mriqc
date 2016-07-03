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
# @Last Modified time: 2016-05-20 09:10:18
""" Encapsulates report generation functions """

import sys
import os
import os.path as op
import collections
import glob
import json

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import jinja2

from mriqc.interfaces.viz_utils import plot_measures, plot_all

# matplotlib.rc('figure', figsize=(11.69, 8.27))  # for DINA4 size
STRUCTURAL_QCGROUPS = [
    ['icvs_csf', 'icvs_gm', 'icvs_wm'],
    ['rpve_csf', 'rpve_gm', 'rpve_wm'],
    ['inu_range', 'inu_med'],
    ['cnr'], ['efc'], ['fber'], ['cjv'],
    ['fwhm_avg', 'fwhm_x', 'fwhm_y', 'fwhm_z'],
    ['qi1', 'qi2'],
    ['snr', 'snr_csf', 'snr_gm', 'snr_wm'],
    ['summary_mean_bg', 'summary_stdv_bg', 'summary_p05_bg', 'summary_p95_bg',
     'summary_mean_csf', 'summary_stdv_csf', 'summary_p05_csf', 'summary_p95_csf',
     'summary_mean_gm', 'summary_stdv_gm', 'summary_p05_gm', 'summary_p95_gm',
     'summary_mean_wm', 'summary_stdv_wm', 'summary_p05_wm', 'summary_p95_wm']
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


def workflow_report(qctype, settings=None):
    """ Creates the report """
    import datetime

    dframe, failed = generate_csv(qctype, settings)
    sub_list = sorted(pd.unique(dframe.subject_id.ravel())) #pylint: disable=E1101

    if qctype == 'anat':
        qctype = 'anatomical'
    elif qctype == 'func':
        qctype = 'functional'

    out_dir = settings.get('output_dir', os.getcwd())
    work_dir = settings.get('work_dir', op.abspath('tmp'))
    out_file = op.join(out_dir, qctype + '_%s.pdf')

    result = {}
    func = getattr(sys.modules[__name__], 'report_' + qctype)

    pdf_group = []
    # Generate summary page
    out_sum = op.join(work_dir, 'summary_group.pdf')
    summary_cover(dframe, qctype, failed=failed, out_file=out_sum)
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
    for subid in sub_list:
        # Get subject-specific info
        subdf = dframe.loc[dframe['subject_id'] == subid]
        sessions = sorted(pd.unique(subdf.session_id.ravel()))
        plots = []
        sess_scans = []
        # Re-build mosaic location
        for sesid in sessions:
            sesdf = subdf.loc[subdf['session_id'] == sesid]
            scans = sorted(pd.unique(sesdf.run_id.ravel()))

            # Each scan has a volume and (optional) fd plot
            for scanid in scans:
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
        if failed:
            sfailed = ['%s (%s)' % (s[1], s[2])
                       for s in failed if subid == s[0]]
        out_sum = op.join(work_dir, '%s_summary_%s.pdf' % (qctype, subid))
        summary_cover(dframe, qctype, failed=sfailed, sub_id=subid, out_file=out_sum)
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


def summary_cover(dframe, qctype, failed=None, sub_id=None, out_file=None):
    """ Generates a cover page with subject information """
    from mriqc import __version__
    import datetime
    import numpy as np
    from rst2pdf.createpdf import RstToPdf
    import pkg_resources as pkgr

    if failed is None:
        failed = []

    newdf = dframe.copy()

    # Format the size
    #pylint: disable=E1101
    newdf[['size_x', 'size_y', 'size_z']] = newdf[['size_x', 'size_y', 'size_z']].astype(np.uint16)
    formatter = lambda row: ur'%d \u00D7 %d \u00D7 %d' % (
        row['size_x'], row['size_y'], row['size_z'])
    newdf['size'] = newdf[['size_x', 'size_y', 'size_z']].apply(formatter, axis=1)

    # Format spacing
    newdf[['spacing_x', 'spacing_y', 'spacing_z']] = newdf[[
        'spacing_x', 'spacing_y', 'spacing_z']].astype(np.float32)  #pylint: disable=E1101
    formatter = lambda row: ur'%.3f \u00D7 %.3f \u00D7 %.3f' % (
        row['spacing_x'], row['spacing_y'], row['spacing_z'])
    newdf['spacing'] = newdf[['spacing_x', 'spacing_y', 'spacing_z']].apply(formatter, axis=1)

    # columns
    cols = ['session_id', 'run_id', 'size', 'spacing']
    colnames = ['Session', 'Run', 'Size', 'Spacing']
    if 'tr' in newdf.columns.ravel():
        cols.append('tr')
        colnames.append('TR (sec)')
    if 'size_t' in newdf.columns.ravel():
        cols.append('size_t')
        colnames.append(r'\# Timepoints')

    # Format parameters table
    if sub_id is None:
        cols.insert(0, 'subject_id')
        colnames.insert(0, 'Subject')
    else:
        newdf = newdf[newdf.subject_id.astype(unicode) == sub_id]

    newdf = newdf[cols]

    colsizes = []
    for col, colname in zip(cols, colnames):
        newdf[[col]] =newdf[[col]].astype(unicode)
        colsize = newdf.loc[:, col].map(len).max()
        colsizes.append(colsize if colsize > len(colname) else len(colname))

    colformat = u' '.join(u'{:<%d}' % c for c in colsizes)
    formatter = lambda row: colformat.format(*row)
    rowsformatted = newdf[cols].apply(formatter, axis=1).ravel().tolist()
    # rowsformatted = [formatter.format(*row) for row in newdf.iterrows()]
    header = colformat.format(*colnames)
    sep = colformat.format(*['=' * c for c in colsizes])
    ptable = '\n'.join([sep, header, sep] + rowsformatted + [sep])

    title = 'MRIQC: %s MRI %s report' % (
        qctype, 'group' if sub_id is None else 'individual')

    # Substitution dictionary
    context = {
        'title': title + '\n' + ''.join(['='] * len(title)),
        'timestamp': datetime.datetime.now().strftime("%Y-%m-%d, %H:%M"),
        'version': __version__,
        'failed': failed,
        'imparams': ptable
    }

    if sub_id is not None:
        context['sub_id'] = sub_id

    if sub_id is None:
        template = ConfigGen(pkgr.resource_filename(
            'mriqc', op.join('data', 'reports', 'cover_group.rst')))
    else:
        template = ConfigGen(pkgr.resource_filename(
            'mriqc', op.join('data', 'reports', 'cover_individual.rst')))

    RstToPdf().createPdf(
        text=template.compile(context), output=out_file)


def concat_pdf(in_files, out_file='concatenated.pdf'):
    """ Concatenate PDF list (http://stackoverflow.com/a/3444735) """
    from PyPDF2 import PdfFileWriter, PdfFileReader

    with open(out_file, 'wb') as out_pdffile:
        outpdf = PdfFileWriter()

        for in_file in in_files:
            with open(in_file, 'rb') as in_pdffile:
                inpdf = PdfFileReader(in_pdffile)
                for fpdf in range(inpdf.numPages):
                    outpdf.addPage(inpdf.getPage(fpdf))
                outpdf.write(out_pdffile)

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
    sessions = sorted(pd.unique(dframe.session_id.ravel()))
    for ssid in sessions:
        sesdf = dframe.copy().loc[dframe['session_id'] == ssid]
        scans = pd.unique(sesdf.run_id.ravel())
        if sc_split:
            for scid in scans:
                subset = sesdf.loc[sesdf['run_id'] == scid]
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

def generate_csv(data_type, settings):
    datalist = []
    errorlist = []
    jsonfiles = glob.glob(op.join(settings['work_dir'], 'derivatives', '%s*.json' % data_type))

    if not jsonfiles:
        raise RuntimeError('No individual QC files were found in the working directory'
                           '\'%s\' for the \'%s\' data type.' % (settings['work_dir'], data_type))

    for jsonfile in jsonfiles:
        dfentry = _read_and_save(jsonfile)
        if dfentry is not None:
            if 'exec_error' not in dfentry.keys():
                datalist.append(dfentry)
            else:
                errorlist.append(dfentry['subject_id'])

    dataframe = pd.DataFrame(datalist)
    cols = dataframe.columns.tolist()  # pylint: disable=no-member

    reorder = []
    for field in ['run', 'session', 'subject']:
        for col in cols:
            if col.startswith(field):
                reorder.append(col)

    for col in reorder:
        cols.remove(col)
        cols.insert(0, col)

    if 'mosaic_file' in cols:
        cols.remove('mosaic_file')

    # Sort the dataframe, with failsafe if pandas version is too old
    try:
        dataframe = dataframe.sort_values(by=['subject_id', 'session_id', 'run_id'])
    except AttributeError:
        #pylint: disable=E1101
        dataframe = dataframe.sort(columns=['subject_id', 'session_id', 'run_id'])

    # Drop duplicates
    try:
        #pylint: disable=E1101
        dataframe.drop_duplicates(['subject_id', 'session_id', 'run_id'], keep='last',
                                  inplace=True)
    except TypeError:
        #pylint: disable=E1101
        dataframe.drop_duplicates(['subject_id', 'session_id', 'run_id'], take_last=True,
                                  inplace=True)

    out_fname = op.join(settings['output_dir'], data_type + 'MRIQC.csv')
    dataframe[cols].to_csv(out_fname, index=False)
    return dataframe, errorlist


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


class ConfigGen(object):
    """
    Utility class for generating a config file from a jinja template.
    https://github.com/oesteban/endofday/blob/f2e79c625d648ef45b08cc1f11fd0bd84342d604/endofday/core/template.py
    """
    def __init__(self, template_str):
        self.template_str = template_str
        self.env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(searchpath='/'),
            trim_blocks=True, lstrip_blocks=True)

    def compile(self, configs):
        template = self.env.get_template(self.template_str)
        return template.render(configs)

    def generate_conf(self, configs, path):
        output = self.compile(configs)
        with open(path, 'w+') as output_file:
            output_file.write(output)
