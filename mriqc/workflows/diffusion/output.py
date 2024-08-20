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
        n_procs=max(1, config.nipype.nprocs // 2),
    )
    mosaic_md = pe.Node(
        PlotMosaic(cmap='Greys_r'),
        name='mosaic_md',
        n_procs=max(1, config.nipype.nprocs // 2),
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
        n_procs=max(1, config.nipype.nprocs // 2),
    )

    mosaic_noise = pe.MapNode(
        PlotMosaic(
            only_noise=True,
            cmap='viridis_r',
        ),
        name='mosaic_noise',
        iterfield=['in_file'],
        n_procs=max(1, config.nipype.nprocs // 2),
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
        n_procs=config.nipype.nprocs,
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
