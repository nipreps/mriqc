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
# @Last Modified time: 2016-09-16 17:58:39
""" Encapsulates report generation functions """
from __future__ import print_function, division, absolute_import, unicode_literals
import sys
import os
import os.path as op
from glob import glob
import logging

from builtins import zip, range, object, str, bytes  # pylint: disable=W0622

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import jinja2

from mriqc.utils.misc import generate_csv
from mriqc.interfaces.viz_utils import (
    plot_mosaic, plot_measures, plot_all, DINA4_LANDSCAPE, DEFAULT_DPI)

STRUCTURAL_QCGROUPS = [
    ['icvs_csf', 'icvs_gm', 'icvs_wm'],
    ['rpve_csf', 'rpve_gm', 'rpve_wm'],
    ['inu_range', 'inu_med'],
    ['cnr'], ['efc'], ['fber'], ['cjv'],
    ['fwhm_avg', 'fwhm_x', 'fwhm_y', 'fwhm_z'],
    ['qi1', 'qi2', 'wm2max'],
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
    ['dvars_std', 'dvars_vstd'],
    ['dvars_nstd'],
    ['fd_mean'],
    ['fd_num'],
    ['fd_perc'],
    ['gcor'],
    ['m_tsnr'],
    ['outlier'],
    ['quality']
]

class MRIQCReportPDF(object):
    """
    Generates group and individual reports
    """

    def __init__(self, qctype, settings, dpi=DEFAULT_DPI, figsize=DINA4_LANDSCAPE):
        if qctype[:4] == 'anat':
            qctype = 'anatomical'
        elif qctype[:4] == 'func':
            qctype = 'functional'
        else:
            raise RuntimeError('Unknown QC data type "{}"'.format(qctype))

        self.qctype = qctype
        self.dpi = dpi
        self.figsize = figsize

        self.out_dir = settings.get('output_dir', os.getcwd())
        self.work_dir = settings.get('work_dir', op.abspath('work'))
        self.report_dir = op.join(self.work_dir, 'reports')

        # Generate csv table
        qcjson = op.join(self.out_dir, 'derivatives', '{}*.json'.format(self.qctype[:4]))
        out_csv = op.join(self.out_dir, qctype[:4] + 'MRIQC.csv')
        self.dataframe, self.failed = generate_csv(glob(qcjson), out_csv)
        self.result = {}

    def group_report(self):
        """ Generates the group report """

        dframe = self.dataframe.copy()
        # Generate summary page
        out_sum = op.join(self.work_dir, 'summary_group.pdf')
        self.summary_cover(out_file=out_sum)
        pdf_group = [out_sum]

        # Generate group report
        qc_group = op.join(self.work_dir, 'qc_measures_group.pdf')
        # Generate violinplots. If successfull, add documentation.
        func = getattr(self, '_report_' + self.qctype)
        func(out_file=qc_group)
        pdf_group += [qc_group]

        if len(pdf_group) > 0:
            out_group_file = op.join(self.out_dir, '%s_group.pdf' % self.qctype)
            # Generate final report with collected pdfs in plots
            concat_pdf(pdf_group, out_group_file)
            self.result['group'] = {'success': True, 'path': out_group_file}

    def _subject_plots(self, subid):
        # Get subject-specific info
        subdf = self.dataframe.loc[self.dataframe['subject_id'] == subid]
        sessions = sorted(pd.unique(subdf.session_id.ravel()))
        subject_plots = []

        # Create figure here to avoid too many figures
        fig = plt.Figure(figsize=DINA4_LANDSCAPE)
        # Re-build mosaic location
        for sesid in sessions:
            sesdf = subdf.loc[subdf['session_id'] == sesid]
            scans = sorted(pd.unique(sesdf.run_id.ravel()))

            # Each scan has a volume and (optional) fd plot
            for scanid in scans:
                nii_paths = op.join(self.report_dir, self.qctype[:4],
                                      '{}_ses-{}_{}/mosaic*.nii.gz'.format(subid, sesid, scanid))
                nii_files = sorted(glob(nii_paths))

                for mosaic in nii_files:
                    fname, ext = op.splitext(op.basename(mosaic))
                    if ext == '.gz':
                        fname, _ = op.splitext(fname)
                    fname = fname[7:]
                    out_mosaic = op.join(
                        self.work_dir, 'mosaic_{}_{}_ses-{}_run-{}_{}.pdf'.format(
                            self.qctype[:4], subid, sesid, scanid, fname))
                    title = 'Filename: {}, session: {}, other: {}'.format(fname, sesid, scanid)
                    fig = plot_mosaic(mosaic, fig=fig, title=title)
                    fig.savefig(out_mosaic, dpi=self.dpi)
                    fig.clf()
                    subject_plots.append(out_mosaic)

                plots = op.join(self.report_dir, self.qctype[:4],
                                '{}_ses-{}_{}/plot_*.pdf'.format(subid, sesid, scanid))

                for fname in sorted(glob(plots)):
                    if op.isfile(fname):
                        subject_plots.append(fname)

        plt.close()
        return subject_plots

    def individual_report(self, sub_list=None):
        if isinstance(sub_list, (str, bytes)):
            sub_list = [sub_list]

        if isinstance(sub_list, tuple):
            sub_list = list(sub_list)

        # Generate all subjects
        if sub_list is None or not sub_list:
            sub_list = sorted(pd.unique(self.dataframe.subject_id.ravel())) #pylint: disable=E1101

        func = getattr(self, '_report_' + self.qctype)

        out_indiv_files = []
        # Generate individual reports for subjects
        for subid in sub_list:
            # Generate all mosaics (mosaic_*.nii.gz)
            plots = self._subject_plots(subid)

            # Summary cover
            sfailed = []
            if self.failed:
                sfailed = ['%s (%s)' % (s[1], s[2])
                           for s in self.failed if subid == s[0]]
            out_sum = op.join(self.work_dir, '%s_summary_%s.pdf' % (self.qctype, subid))
            self.summary_cover(sub_id=subid, out_file=out_sum)
            plots.insert(0, out_sum)

            # Summary (violinplots) of QC measures
            qc_ms = op.join(self.work_dir, '%s_measures_%s.pdf' % (self.qctype, subid))

            func(subject=subid, out_file=qc_ms)
            plots.append(qc_ms)

            if len(plots) > 0:
                # Generate final report with collected pdfs in plots
                sub_path = op.join(self.out_dir, '{}_{}.pdf'.format(self.qctype, subid))
                concat_pdf(plots, sub_path)
                out_indiv_files.append(sub_path)
                self.result[subid] = {'success': True, 'path': sub_path}

        return out_indiv_files


    def _report_anatomical(
            self, subject=None, sc_split=False, condensed=True,
            out_file='anatomical.pdf'):
        """ Calls the report generator on the functional measures """
        return _write_report(
            self.dataframe, STRUCTURAL_QCGROUPS, sub_id=subject, sc_split=sc_split,
            condensed=condensed, out_file=out_file)

    def _report_functional(
            self, subject=None, sc_split=False, condensed=True,
            out_file='functional.pdf'):
        """ Calls the report generator on the functional measures """
        from tempfile import mkdtemp

        wdir = mkdtemp()
        fspatial = _write_report(
            self.dataframe, FUNC_TEMPORAL_QCGROUPS, sub_id=subject, sc_split=sc_split,
            condensed=condensed, out_file=op.join(wdir, 'fspatial.pdf'))

        ftemporal = _write_report(
            self.dataframe, FUNC_SPATIAL_QCGROUPS, sub_id=subject, sc_split=sc_split,
            condensed=condensed, out_file=op.join(wdir, 'ftemporal.pdf'))

        concat_pdf([fspatial, ftemporal], out_file)
        return out_file

    def summary_cover(self, sub_id=None, out_file=None):
        """ Generates a cover page with subject information """
        from mriqc import __version__
        import datetime
        import numpy as np
        from logging import CRITICAL
        from rst2pdf.createpdf import RstToPdf
        from rst2pdf.log import log
        import pkg_resources as pkgr

        failed = self.failed
        if failed is None:
            failed = []

        log.setLevel(CRITICAL)
        newdf = self.dataframe.copy()

        # Format the size
        #pylint: disable=E1101
        newdf[['size_x', 'size_y', 'size_z']] = newdf[['size_x', 'size_y', 'size_z']].astype(np.uint16)
        formatter = lambda row: '%d \u00D7 %d \u00D7 %d' % (
            row['size_x'], row['size_y'], row['size_z'])
        newdf['size'] = newdf[['size_x', 'size_y', 'size_z']].apply(formatter, axis=1)

        # Format spacing
        newdf[['spacing_x', 'spacing_y', 'spacing_z']] = newdf[[
            'spacing_x', 'spacing_y', 'spacing_z']].astype(np.float32)  #pylint: disable=E1101
        formatter = lambda row: '%.3f \u00D7 %.3f \u00D7 %.3f' % (
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
            colnames.append('# Timepoints')

        # Format parameters table
        if sub_id is None:
            cols.insert(0, 'subject_id')
            colnames.insert(0, 'Subject')
        else:
            newdf = newdf[newdf.subject_id == sub_id]

        newdf = newdf[cols]

        colsizes = []
        for col, colname in zip(cols, colnames):
            try:
                newdf[[col]] = newdf[[col]].astype(str)
            except NameError:
                newdf[[col]] = newdf[[col]].astype(str)

            colsize = np.max([len('{}'.format(val)) for val in newdf.loc[:, col]])
            # colsize = newdf.loc[:, col].map(len).max()
            colsizes.append(colsize if colsize > len(colname) else len(colname))

        colformat = ' '.join('{:<%d}' % c for c in colsizes)
        formatter = lambda row: colformat.format(*row)
        rowsformatted = newdf[cols].apply(formatter, axis=1).ravel().tolist()
        # rowsformatted = [formatter.format(*row) for row in newdf.iterrows()]
        header = colformat.format(*colnames)
        sep = colformat.format(*['=' * c for c in colsizes])
        ptable = '\n'.join([sep, header, sep] + rowsformatted + [sep])

        title = 'MRIQC: %s MRI %s report' % (
            self.qctype, 'group' if sub_id is None else 'individual')

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
                  out_file='report.pdf', dpi=DEFAULT_DPI):
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
                        subtitle = '(session: %s other: %s)' % (ssid, scid)
                    else:
                        subtitle = '(Subject: %s, session: %s, other: %s)' % (sub_id, ssid, scid)
                    if condensed:
                        fig = plot_all(sesdf, groups, subject=sub_id,
                                       title='QC measures ' + subtitle)
                    else:
                        fig = plot_measures(
                            sesdf, headers, subject=sub_id,
                            title='QC measures ' + subtitle)
                    report.savefig(fig, dpi=dpi)
                    fig.clf()
        else:
            if len(sesdf.index) > 1:
                if sub_id is None:
                    subtitle = '(session %s)' % (ssid)
                else:
                    subtitle = '(subject %s, session %s)' % (sub_id, ssid)
                if condensed:
                    fig = plot_all(sesdf, groups, subject=sub_id,
                                   title='QC measures ' + subtitle)
                else:
                    fig = plot_measures(
                        sesdf, headers, subject=sub_id,
                        title='QC measures ' + subtitle)
                report.savefig(fig, dpi=dpi)
                fig.clf()

    report.close()
    plt.close()
    # print 'Written report file %s' % out_file
    return out_file


class ConfigGen(object):
    """
    Utility class for generating a config file from a jinja template.
    https://github.com/oesteban/endofday/blob/f2e79c625d648ef45b08cc1\
f11fd0bd84342d604/endofday/core/template.py
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
