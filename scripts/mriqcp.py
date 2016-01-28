#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: oesteban
# @Date:   2015-11-19 16:44:27
# @Last Modified by:   oesteban
# @Last Modified time: 2016-01-18 14:39:54

"""
=====
MRIQC
=====
"""
import os
import os.path as op
import sys

__author__ = "Oscar Esteban"
__copyright__ = ("Copyright 2016, Center for Reproducible Neuroscience, "
                 "Stanford University")
__credits__ = "Oscar Esteban"
__license__ = "BSD"
__version__ = "0.0.1"
__maintainer__ = "Oscar Esteban"
__email__ = "code@oscaresteban.es"
__status__ = "Prototype"


if __name__ == '__main__':
    from argparse import ArgumentParser
    from argparse import RawTextHelpFormatter
    from mriqc.workflows import anat_qc_workflow, fmri_qc_workflow
    from mriqc.utils import gather_bids_data, reorder_csv
    from mriqc.reports import workflow_report
    from nipype import config as ncfg

    parser = ArgumentParser(description='MRI Quality Control',
                            formatter_class=RawTextHelpFormatter)

    g_input = parser.add_argument_group('Inputs')
    g_input.add_argument('-i', '--bids-root', action='store',
                         default=os.getcwd())
    g_input.add_argument('--nthreads', action='store', default=0,
                         type=int, help='number of repetitions')
    g_input.add_argument(
        "--write-graph", action='store_true', default=False,
        help="Write workflow graph.")
    g_input.add_argument(
        "--use-plugin", action='store', default=None,
        help='nipype plugin configuration file')

    g_input.add_argument(
        "--skip-anatomical", action='store_true', default=False,
        help="Skip anatomical QC workflow.")
    g_input.add_argument(
        "--skip-functional", action='store_true', default=False,
        help="Skip functional QC workflow.")

    g_outputs = parser.add_argument_group('Outputs')
    g_outputs.add_argument('-o', '--output-dir', action='store')
    g_outputs.add_argument('-w', '--work-dir', action='store')

    opts = parser.parse_args()

    settings = {'bids_root': op.abspath(opts.bids_root)}

    settings['output_dir'] = os.getcwd()
    if opts.output_dir:
        settings['output_dir'] = op.abspath(opts.output_dir)

    if not op.exists(settings['output_dir']):
        os.makedirs(settings['output_dir'])

    if opts.work_dir:
        settings['work_dir'] = op.abspath(opts.work_dir)

        log_dir = op.join(settings['work_dir'], 'log')
        if not op.exists(log_dir):
            os.makedirs(log_dir)

        # Set nipype config
        ncfg.update_config({
            'logging': {'log_directory': log_dir, 'log_to_file': True},
            'execution': {'crashdump_dir': log_dir}
        })

    plugin_settings = {'plugin': 'Linear'}
    if opts.use_plugin is not None:
        from yaml import load as loadyml
        with open(opts.use_plugin) as f:
            plugin_settings = loadyml(f)
    else:
        # Setup multiprocessing
        nthreads = opts.nthreads
        if nthreads == 0:
            from multiprocessing import cpu_count
            nthreads = cpu_count()

        settings['nthreads'] = nthreads

        if nthreads > 1:
            plugin_settings['plugin'] = 'MultiProc'
            plugin_settings['plugin_args'] = {'n_proc': nthreads, 'maxtasksperchild': 4}

    subjects = gather_bids_data(settings['bids_root'])

    if not any([len(subjects[k])>0 for k in subjects.keys()]):
        raise RuntimeError('No scans found in %s' % settings['bids_root'])

    if not opts.skip_anatomical and subjects['anat']:
        anat_wf, out_csv = anat_qc_workflow(sub_list=subjects['anat'],
                                            settings=settings)

        if opts.write_graph:
            anat_wf.write_graph()

        anat_wf.run(**plugin_settings)
        reports = workflow_report(out_csv, 'anatomical', sub_list=subjects['anat'], settings=settings)
        reorder_csv(out_csv)

    if not opts.skip_functional and subjects['func']:
        func_wf, out_csv = fmri_qc_workflow(sub_list=subjects['func'],
                                            settings=settings)

        if opts.write_graph:
            func_wf.write_graph()

        func_wf.run(**plugin_settings)
        reports = workflow_report(out_csv, 'functional', sub_list=subjects['func'], settings=settings)
        reorder_csv(out_csv)
