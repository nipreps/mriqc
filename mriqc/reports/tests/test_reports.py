#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# @Author: oesteban
# @Date:   2016-01-05 11:33:39
# @Email:  code@oscaresteban.es
"""Test utils"""
# import os.path as op
# import pytest
# from io import open # pylint: disable=W0622
# from mriqc.reports.utils import check_reports


# @pytest.fixture(scope='session')
# @pytest.mark.parametrize("dataset,reports,expected", [
#     ({'t1w': ['/data/bidsroot/sub-06/anat/sub-06_T1w.nii.gz',
#               '/data/bidsroot/sub-11/anat/sub-11_T1w.nii.gz',
#               '/data/bidsroot/sub-05/anat/sub-05_T1w.nii.gz',
#               '/data/bidsroot/sub-13/anat/sub-13_T1w.nii.gz'],
#       'func': ['/data/bidsroot/sub-09/func/sub-09_task-rhymejudgment_bold.nii.gz',
#                   '/data/bidsroot/sub-03/func/sub-03_task-rhymejudgment_bold.nii.gz',
#                   '/data/bidsroot/sub-06/func/sub-06_task-rhymejudgment_bold.nii.gz']},
#      ['anatomical_sub-05_ses-default-session_run-default-run_report.html',
#       'anatomical_sub-06_ses-default-session_run-default-run_report.html',
#       'anatomical_sub-13_ses-default-session_run-default-run_report.html',
#       'functional_sub-06_ses-default_session_task-rhymejudgment_run-default_run_report.html'])
# ])
# def test_check_reports(tmpdir_factory, dataset, reports):
#     out_folder = tmpdir_factory.mktemp('reports')
#     for rname in reports:
#         open(op.join(out_folder, rname), 'a').close()
#     settings = {'reports_dir': out_folder,
#                 'bids_dir': '/data/bidsroot'}

#     check_reports(dataset, settings)
#     assert True
