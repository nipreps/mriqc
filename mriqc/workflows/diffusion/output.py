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
"""Writing out diffusion reportlets."""
from nipype.interfaces import utility as niu
from nipype.pipeline import engine as pe
from nireports.interfaces.dmri import DWIHeatmap
from nireports.interfaces.reporting.base import (
    SimpleBeforeAfterRPT as SimpleBeforeAfter,
)

from mriqc import config
from mriqc.interfaces import DerivativesDataSink


def init_dwi_report_wf(name='dwi_report_wf'):
    """
    Write out individual reportlets.

    .. workflow::

        from mriqc.workflows.diffusion.output import init_dwi_report_wf
        from mriqc.testing import mock_config
        with mock_config():
            wf = init_dwi_report_wf()

    """
    from nireports.interfaces import FMRISummary, PlotMosaic, PlotSpikes
    from niworkflows.interfaces.morphology import BinaryDilation, BinarySubtraction

    # from mriqc.interfaces.reports import IndividualReport

    verbose = config.execution.verbose_reports
    mem_gb = config.workflow.biggest_file_gb
    reportlets_dir = config.execution.work_dir / 'reportlets'

    workflow = pe.Workflow(name=name)
    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                'in_epi',
                'brain_mask',
                'cc_mask',
                'in_avgmap',
                'in_stdmap',
                'in_shells',
                'in_fa',
                'in_md',
                'in_parcellation',
                'in_bdict',
                'noise_floor',
                'name_source',
            ]
        ),
        name='inputnode',
    )

    # Set FD threshold
    # inputnode.inputs.fd_thres = config.workflow.fd_thres

    mosaic_fa = pe.Node(
        PlotMosaic(cmap='Greys_r'),
        name='mosaic_fa',
    )
    mosaic_md = pe.Node(
        PlotMosaic(cmap='Greys_r'),
        name='mosaic_md',
    )

    mosaic_snr = pe.MapNode(
        SimpleBeforeAfter(
            fixed_params={'cmap': 'viridis'},
            moving_params={'cmap': 'Greys_r'},
            before_label='Average',
            after_label='Standard Deviation',
            dismiss_affine=True,
        ),
        name='mosaic_snr',
        iterfield=['before', 'after'],
    )

    mosaic_noise = pe.MapNode(
        PlotMosaic(
            only_noise=True,
            cmap='viridis_r',
        ),
        name='mosaic_noise',
        iterfield=['in_file'],
    )

    if config.workflow.species.lower() in ('rat', 'mouse'):
        mosaic_noise.inputs.view = ['coronal', 'axial']
        mosaic_fa.inputs.view = ['coronal', 'axial']
        mosaic_md.inputs.view = ['coronal', 'axial']

    ds_report_snr = pe.MapNode(
        DerivativesDataSink(
            base_directory=reportlets_dir,
            desc='avgstd',
            datatype='figures',
            allowed_entities=('bval',),
        ),
        name='ds_report_snr',
        run_without_submitting=True,
        iterfield=['in_file', 'bval'],
    )

    ds_report_noise = pe.MapNode(
        DerivativesDataSink(
            base_directory=reportlets_dir,
            desc='background',
            datatype='figures',
            allowed_entities=('bval',),
        ),
        name='ds_report_noise',
        run_without_submitting=True,
        iterfield=['in_file', 'bval'],
    )

    ds_report_fa = pe.Node(
        DerivativesDataSink(
            base_directory=reportlets_dir,
            desc='fa',
            datatype='figures',
        ),
        name='ds_report_fa',
        run_without_submitting=True,
    )

    ds_report_md = pe.Node(
        DerivativesDataSink(
            base_directory=reportlets_dir,
            desc='md',
            datatype='figures',
        ),
        name='ds_report_md',
        run_without_submitting=True,
    )

    def _gen_entity(inlist):
        return ['00000'] + [f'{int(round(bval, 0)):05d}' for bval in inlist]

    # fmt: off
    workflow.connect([
        (inputnode, mosaic_snr, [('in_avgmap', 'before'),
                                 ('in_stdmap', 'after'),
                                 ('brain_mask', 'wm_seg')]),
        (inputnode, mosaic_noise, [('in_avgmap', 'in_file')]),
        (inputnode, mosaic_fa, [('in_fa', 'in_file'),
                                ('brain_mask', 'bbox_mask_file')]),
        (inputnode, mosaic_md, [('in_md', 'in_file'),
                                ('brain_mask', 'bbox_mask_file')]),
        (inputnode, ds_report_snr, [('name_source', 'source_file'),
                                    (('in_shells', _gen_entity), 'bval')]),
        (inputnode, ds_report_noise, [('name_source', 'source_file'),
                                      (('in_shells', _gen_entity), 'bval')]),
        (inputnode, ds_report_fa, [('name_source', 'source_file')]),
        (inputnode, ds_report_md, [('name_source', 'source_file')]),
        (mosaic_snr, ds_report_snr, [('out_report', 'in_file')]),
        (mosaic_noise, ds_report_noise, [('out_file', 'in_file')]),
        (mosaic_fa, ds_report_fa, [('out_file', 'in_file')]),
        (mosaic_md, ds_report_md, [('out_file', 'in_file')]),
    ])
    # fmt: on

    get_wm = pe.Node(niu.Function(function=_get_wm), name='get_wm')
    plot_heatmap = pe.Node(
        DWIHeatmap(scalarmap_label='Shell-wise Fractional Anisotropy (FA)'),
        name='plot_heatmap',
    )
    ds_report_hm = pe.Node(
        DerivativesDataSink(
            base_directory=reportlets_dir,
            desc='heatmap',
            datatype='figures',
        ),
        name='ds_report_hm',
        run_without_submitting=True,
    )

    # fmt: off
    workflow.connect([
        (inputnode, get_wm, [('in_parcellation', 'in_file')]),
        (inputnode, plot_heatmap, [('in_epi', 'in_file'),
                                   ('in_fa', 'scalarmap'),
                                   ('in_bdict', 'b_indices'),
                                   ('noise_floor', 'sigma')]),
        (inputnode, ds_report_hm, [('name_source', 'source_file')]),
        (get_wm, plot_heatmap, [('out', 'mask_file')]),
        (plot_heatmap, ds_report_hm, [('out_file', 'in_file')]),

    ])
    # fmt: on

    if True:
        return workflow

    # Generate crown mask
    # Create the crown mask
    dilated_mask = pe.Node(BinaryDilation(), name='dilated_mask')
    subtract_mask = pe.Node(BinarySubtraction(), name='subtract_mask')
    parcels = pe.Node(niu.Function(function=_carpet_parcellation), name='parcels')

    bigplot = pe.Node(FMRISummary(), name='BigPlot', mem_gb=mem_gb * 3.5)

    ds_report_carpet = pe.Node(
        DerivativesDataSink(
            base_directory=reportlets_dir,
            desc='carpet',
            datatype='figures',
        ),
        name='ds_report_carpet',
        run_without_submitting=True,
    )

    # fmt: off
    workflow.connect([
        # (inputnode, rnode, [("in_iqms", "in_iqms")]),
        (inputnode, bigplot, [('hmc_epi', 'in_func'),
                              ('hmc_fd', 'fd'),
                              ('fd_thres', 'fd_thres'),
                              ('in_dvars', 'dvars'),
                              ('outliers', 'outliers'),
                              (('meta_sidecar', _get_tr), 'tr')]),
        (inputnode, parcels, [('epi_parc', 'segmentation')]),
        (inputnode, dilated_mask, [('brain_mask', 'in_mask')]),
        (inputnode, subtract_mask, [('brain_mask', 'in_subtract')]),
        (dilated_mask, subtract_mask, [('out_mask', 'in_base')]),
        (subtract_mask, parcels, [('out_mask', 'crown_mask')]),
        (parcels, bigplot, [('out', 'in_segm')]),
        (inputnode, ds_report_carpet, [('name_source', 'source_file')]),
        (bigplot, ds_report_carpet, [('out_file', 'in_file')]),
    ])
    # fmt: on

    if config.workflow.fft_spikes_detector:
        mosaic_spikes = pe.Node(
            PlotSpikes(
                out_file='plot_spikes.svg',
                cmap='viridis',
                title='High-Frequency spikes',
            ),
            name='PlotSpikes',
        )

        ds_report_spikes = pe.Node(
            DerivativesDataSink(
                base_directory=reportlets_dir,
                desc='spikes',
                datatype='figures',
            ),
            name='ds_report_spikes',
            run_without_submitting=True,
        )

        # fmt: off
        workflow.connect([
            (inputnode, ds_report_spikes, [('name_source', 'source_file')]),
            (inputnode, mosaic_spikes, [('in_ras', 'in_file'),
                                        ('in_spikes', 'in_spikes'),
                                        ('in_fft', 'in_fft')]),
            (mosaic_spikes, ds_report_spikes, [('out_file', 'in_file')]),
        ])
        # fmt: on

    if not verbose:
        return workflow

    # Verbose-reporting goes here
    from nireports.interfaces import PlotContours

    mosaic_zoom = pe.Node(
        PlotMosaic(
            cmap='Greys_r',
        ),
        name='PlotMosaicZoomed',
    )

    plot_bmask = pe.Node(
        PlotContours(
            display_mode='y' if config.workflow.species.lower() in ('rat', 'mouse') else 'z',
            levels=[0.5],
            colors=['r'],
            cut_coords=10,
            out_file='bmask',
        ),
        name='PlotBrainmask',
    )

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

    ds_report_bmask = pe.Node(
        DerivativesDataSink(
            base_directory=reportlets_dir,
            desc='brainmask',
            datatype='figures',
        ),
        name='ds_report_bmask',
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

    # fmt: off
    workflow.connect([
        (inputnode, ds_report_norm, [('mni_report', 'in_file'),
                                     ('name_source', 'source_file')]),
        (inputnode, plot_bmask, [('epi_mean', 'in_file'),
                                 ('brain_mask', 'in_contours')]),
        (inputnode, mosaic_zoom, [('epi_mean', 'in_file'),
                                  ('brain_mask', 'bbox_mask_file')]),
        (inputnode, mosaic_noise, [('epi_mean', 'in_file')]),
        (inputnode, ds_report_zoomed, [('name_source', 'source_file')]),
        (inputnode, ds_report_background, [('name_source', 'source_file')]),
        (inputnode, ds_report_bmask, [('name_source', 'source_file')]),
        (mosaic_zoom, ds_report_zoomed, [('out_file', 'in_file')]),
        (mosaic_noise, ds_report_background, [('out_file', 'in_file')]),
        (plot_bmask, ds_report_bmask, [('out_file', 'in_file')]),
    ])
    # fmt: on

    return workflow


def _carpet_parcellation(segmentation, crown_mask):
    """Generate the union of two masks."""
    from pathlib import Path

    import nibabel as nb
    import numpy as np

    img = nb.load(segmentation)

    lut = np.zeros((256,), dtype='uint8')
    lut[100:201] = 1  # Ctx GM
    lut[30:99] = 2  # dGM
    lut[1:11] = 3  # WM+CSF
    lut[255] = 4  # Cerebellum
    # Apply lookup table
    seg = lut[np.asanyarray(img.dataobj, dtype='uint16')]
    seg[np.asanyarray(nb.load(crown_mask).dataobj, dtype=int) > 0] = 5

    outimg = img.__class__(seg.astype('uint8'), img.affine, img.header)
    outimg.set_data_dtype('uint8')
    out_file = Path('segments.nii.gz').absolute()
    outimg.to_filename(out_file)
    return str(out_file)


def _get_tr(meta_dict):
    return meta_dict.get('RepetitionTime', None)


def _get_wm(in_file, radius=2):
    from pathlib import Path

    import nibabel as nb
    import numpy as np
    from nipype.utils.filemanip import fname_presuffix
    from scipy import ndimage as ndi
    from skimage.morphology import ball

    parc = nb.load(in_file)
    hdr = parc.header.copy()
    data = np.array(parc.dataobj, dtype=hdr.get_data_dtype())
    wm_mask = ndi.binary_erosion((data == 1) | (data == 2), ball(radius))

    hdr.set_data_dtype(np.uint8)
    out_wm = fname_presuffix(in_file, suffix='wm', newpath=str(Path.cwd()))
    parc.__class__(
        wm_mask.astype(np.uint8),
        parc.affine,
        hdr,
    ).to_filename(out_wm)
    return out_wm
