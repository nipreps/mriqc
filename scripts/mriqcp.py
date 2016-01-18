#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: oesteban
# @Date:   2015-11-19 16:44:27
# @Last Modified by:   oesteban
# @Last Modified time: 2016-01-18 08:15:33

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
    from mriqc.workflows import anat_qc_workflow
    from mriqc.utils import gather_bids_data
    from mriqc.reports import workflow_report

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

    # Setup multiprocessing
    nthreads = opts.nthreads
    if nthreads == 0:
        from multiprocessing import cpu_count
        nthreads = cpu_count()

    settings['nthreads'] = nthreads

    plugin = 'Linear'
    plugin_args = {}
    if nthreads > 1:
        plugin = 'MultiProc'
        plugin_args = {'n_proc': nthreads, 'maxtasksperchild': 4}

    subjects = gather_bids_data(settings['bids_root'])

    if subjects['anat']:
        anat_wf, out_csv = anat_qc_workflow(sub_list=subjects['anat'],
                                            settings=settings)

        if opts.write_graph:
            anat_wf.write_graph()

        anat_wf.run()
        reports = workflow_report(out_csv, 'anatomical', settings=settings)

    if subjects['func']:
        func_wf, out_csv = func_qc_workflow(sub_list=subjects['func'],
                                            settings=settings)

        if opts.write_graph:
            func_wf.write_graph()

        func_wf.run()
        reports = workflow_report(out_csv, 'functional', settings=settings)
