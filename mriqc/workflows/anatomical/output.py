# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# Copyright 2023 The NiPreps Developers <nipreps@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# We support and encourage derived works from this project, please read
# about our expectations at
#
#     https://www.nipreps.org/community/licensing/
#
"""Writing out anatomical reportlets."""

from nipype.interfaces import utility as niu
from nipype.pipeline import engine as pe

from mriqc import config
from mriqc.interfaces import DerivativesDataSink


def init_anat_report_wf(name: str = 'anat_report_wf'):
    """
    Generate the components of the individual report.

    .. workflow::

        from mriqc.workflows.anatomical.output import init_anat_report_wf
        from mriqc.testing import mock_config
        with mock_config():
            wf = init_anat_report_wf()

    """
    from nireports.interfaces import PlotMosaic

    # from mriqc.interfaces.reports import IndividualReport

    verbose = config.execution.verbose_reports
    reportlets_dir = config.execution.work_dir / 'reportlets'

    workflow = pe.Workflow(name=name)
    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                'in_ras',
                'brainmask',
                'headmask',
                'airmask',
                'artmask',
                'rotmask',
                'segmentation',
                'inu_corrected',
                'noisefit',
                'in_iqms',
                'mni_report',
                'api_id',
                'name_source',
            ]
        ),
        name='inputnode',
    )

    mosaic_zoom = pe.Node(
        PlotMosaic(cmap='Greys_r'),
        name='PlotMosaicZoomed',
    )

    mosaic_noise = pe.Node(
        PlotMosaic(only_noise=True, cmap='viridis_r'),
        name='PlotMosaicNoise',
    )
    if config.workflow.species.lower() in ('rat', 'mouse'):
        mosaic_zoom.inputs.view = ['coronal', 'axial']
        mosaic_noise.inputs.view = ['coronal', 'axial']

    ds_report_zoomed = pe.Node(
        DerivativesDataSink(
            base_directory=reportlets_dir,
            desc='zoomed',
            datatype='figures',
        ),
        name='ds_report_zoomed',
        run_without_submitting=True,
    )

    ds_report_background = pe.Node(
        DerivativesDataSink(
            base_directory=reportlets_dir,
            desc='background',
            datatype='figures',
        ),
        name='ds_report_background',
        run_without_submitting=True,
    )

    # fmt: off
    workflow.connect([
        # (inputnode, rnode, [("in_iqms", "in_iqms")]),
        (inputnode, mosaic_zoom, [('in_ras', 'in_file'),
                                  ('brainmask', 'bbox_mask_file')]),
        (inputnode, mosaic_noise, [('in_ras', 'in_file')]),
        (inputnode, ds_report_zoomed, [('name_source', 'source_file')]),
        (inputnode, ds_report_background, [('name_source', 'source_file')]),
        (mosaic_zoom, ds_report_zoomed, [('out_file', 'in_file')]),
        (mosaic_noise, ds_report_background, [('out_file', 'in_file')]),
    ])
    # fmt: on

    if not verbose:
        return workflow

    from nireports.interfaces import PlotContours

    display_mode = 'y' if config.workflow.species.lower() in ('rat', 'mouse') else 'z'
    plot_segm = pe.Node(
        PlotContours(
            display_mode=display_mode,
            levels=[0.5, 1.5, 2.5],
            cut_coords=10,
            colors=['r', 'g', 'b'],
        ),
        name='PlotSegmentation',
    )

    ds_report_segm = pe.Node(
        DerivativesDataSink(
            base_directory=reportlets_dir,
            desc='segmentation',
            datatype='figures',
        ),
        name='ds_report_segm',
        run_without_submitting=True,
    )

    plot_bmask = pe.Node(
        PlotContours(
            display_mode=display_mode,
            levels=[0.5],
            colors=['r'],
            cut_coords=10,
            out_file='bmask',
        ),
        name='PlotBrainmask',
    )

    ds_report_bmask = pe.Node(
        DerivativesDataSink(
            base_directory=reportlets_dir,
            desc='brainmask',
            datatype='figures',
        ),
        name='ds_report_bmask',
        run_without_submitting=True,
    )

    plot_artmask = pe.Node(
        PlotContours(
            display_mode=display_mode,
            levels=[0.5],
            colors=['r'],
            cut_coords=10,
            out_file='artmask',
            saturate=True,
        ),
        name='PlotArtmask',
    )

    ds_report_artmask = pe.Node(
        DerivativesDataSink(
            base_directory=reportlets_dir,
            desc='artifacts',
            datatype='figures',
        ),
        name='ds_report_artmask',
        run_without_submitting=True,
    )

    # NOTE: humans switch on these two to coronal view.
    display_mode = 'y' if config.workflow.species.lower() in ('rat', 'mouse') else 'x'
    plot_airmask = pe.Node(
        PlotContours(
            display_mode=display_mode,
            levels=[0.5],
            colors=['r'],
            cut_coords=6,
            out_file='airmask',
        ),
        name='PlotAirmask',
    )

    ds_report_airmask = pe.Node(
        DerivativesDataSink(
            base_directory=reportlets_dir,
            desc='airmask',
            datatype='figures',
        ),
        name='ds_report_airmask',
        run_without_submitting=True,
    )

    plot_headmask = pe.Node(
        PlotContours(
            display_mode=display_mode,
            levels=[0.5],
            colors=['r'],
            cut_coords=6,
            out_file='headmask',
        ),
        name='PlotHeadmask',
    )

    ds_report_headmask = pe.Node(
        DerivativesDataSink(
            base_directory=reportlets_dir,
            desc='head',
            datatype='figures',
        ),
        name='ds_report_headmask',
        run_without_submitting=True,
    )

    ds_report_norm = pe.Node(
        DerivativesDataSink(
            base_directory=reportlets_dir,
            desc='norm',
            datatype='figures',
        ),
        name='ds_report_norm',
        run_without_submitting=True,
    )

    ds_report_noisefit = pe.Node(
        DerivativesDataSink(
            base_directory=reportlets_dir,
            desc='noisefit',
            datatype='figures',
        ),
        name='ds_report_noisefit',
        run_without_submitting=True,
    )

    # fmt: off
    workflow.connect([
        (inputnode, ds_report_segm, [('name_source', 'source_file')]),
        (inputnode, ds_report_bmask, [('name_source', 'source_file')]),
        (inputnode, ds_report_artmask, [('name_source', 'source_file')]),
        (inputnode, ds_report_airmask, [('name_source', 'source_file')]),
        (inputnode, ds_report_headmask, [('name_source', 'source_file')]),
        (inputnode, ds_report_norm, [('mni_report', 'in_file'),
                                     ('name_source', 'source_file')]),
        (inputnode, ds_report_noisefit, [('noisefit', 'in_file'),
                                         ('name_source', 'source_file')]),
        (inputnode, plot_segm, [('in_ras', 'in_file'),
                                ('segmentation', 'in_contours')]),
        (inputnode, plot_bmask, [('in_ras', 'in_file'),
                                 ('brainmask', 'in_contours')]),
        (inputnode, plot_headmask, [('in_ras', 'in_file'),
                                    ('headmask', 'in_contours')]),
        (inputnode, plot_airmask, [('in_ras', 'in_file'),
                                   ('airmask', 'in_contours')]),
        (inputnode, plot_artmask, [('in_ras', 'in_file'),
                                   ('artmask', 'in_contours')]),
        (plot_bmask, ds_report_bmask, [('out_file', 'in_file')]),
        (plot_segm, ds_report_segm, [('out_file', 'in_file')]),
        (plot_artmask, ds_report_artmask, [('out_file', 'in_file')]),
        (plot_headmask, ds_report_headmask, [('out_file', 'in_file')]),
        (plot_airmask, ds_report_airmask, [('out_file', 'in_file')]),
    ])
    # fmt: on

    return workflow
