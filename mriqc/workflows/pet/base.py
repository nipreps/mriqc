# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# Copyright 2021 The NiPreps Developers <nipreps@gmail.com>
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
"""Workflows building blocks used to generate PET QC derivatives."""
import os.path as op

from nilearn.plotting import plot_carpet
from nipype.interfaces import utility as niu
from nipype.interfaces.ants import ApplyTransforms
from nipype.interfaces.utility import Function, IdentityInterface
from nipype.pipeline import engine as pe
from niworkflows.interfaces.bids import ReadSidecarJSON
from niworkflows.interfaces.reportlets.registration import (
    SpatialNormalizationRPT as RobustMNINormalization,
)
from pkg_resources import resource_filename
from mriqc import config
from mriqc.interfaces import DerivativesDataSink
from mriqc.workflows.pet.output import init_pet_report_wf


def pet_qc_workflow(name='petMRIQC'):
    """
    Initialize the (pet)MRIQC workflow.

    .. workflow::

        import os.path as op
        from mriqc.workflows.functional.base import pet_qc_workflow
        from mriqc.testing import mock_config
        with mock_config():
            wf = pet_qc_workflow()

    """

    from nipype.interfaces.afni import TStat
    from mriqc.messages import BUILDING_WORKFLOW

    dataset = config.workflow.inputs['pet']
    metadata = config.workflow.inputs_metadata['pet']
    entities = config.workflow.inputs_entities['pet']

    message = BUILDING_WORKFLOW.format(
        modality='pet',
        detail=f'for {len(dataset)} PET runs.',
    )
    config.loggers.workflow.info(message)

    workflow = pe.Workflow(name=name)

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=['in_file', 'metadata', 'entities'],
        ),
        name='inputnode',
    )
    inputnode.synchronize = True
    inputnode.iterables = [
        ('in_file', dataset),
        ('metadata', metadata),
        ('entities', entities),
    ]

    load_meta = pe.Node(ReadSidecarJSON(bids_dir=config.execution.bids_dir), name='LoadMetadata')

    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                'qc',
                'mosaic',
                'out_group',
                'out_fd',
                'pet_mean',
                'pet_dseg',
                'tacs_tsv',
                'norm_report',
                'tacs_figures',
            ]
        ),
        name='outputnode',
    )

    hmcwf = hmc(omp_nthreads=config.nipype.omp_nthreads)
    hmcwf.inputs.inputnode.fd_radius = config.workflow.fd_radius

    mean_pet = pe.Node(TStat(args='-mean', outputtype='NIFTI_GZ'), name='MeanPET')

    normwf = pet_mni_align()
    tacswf = extract_tacs()
    iqmswf = compute_iqms()
    pet_report_wf = init_pet_report_wf()

    norm_report_sink = pe.Node(
        DerivativesDataSink(
            base_directory=config.execution.work_dir / 'reportlets',
            datatype='figures',
            desc='norm',
            extension='.svg',
            dismiss_entities=('part',)
        ),
        name='norm_report_sink',
        run_without_submitting=True,
    )

    carpet_plot = pe.Node(Function(
        input_names=['in_pet', 'seg_file', 'metadata', 'output_file'],
        output_names=['out_file'],
        function=create_pet_carpet_plot
    ), name='carpet_plot')

    carpet_plot.inputs.output_file = 'carpet_plot.svg'

    # DataSink node for carpet plot
    ds_report_carpet = pe.Node(
        DerivativesDataSink(
            base_directory=config.execution.work_dir / 'reportlets',
            datatype='figures',
            desc='carpet',
            extension='.svg',
            dismiss_entities=('part',),
        ),
        name='ds_report_carpet',
        run_without_submitting=True,
    )

    ds_tacs = pe.MapNode(
        DerivativesDataSink(
            base_directory=str(config.execution.output_dir),
            suffix='timeseries',
            atlas="hammers",
            space="MNI152",
            datatype='pet',
            dismiss_entities=("desc",),
            extension='.tsv',
        ),
        name='ds_tacs',
        run_without_submitting=True,
        iterfield=['in_file', 'source_file'],
    )

    workflow.connect([
        (inputnode, load_meta, [('in_file', 'in_file')]),
        (inputnode, hmcwf, [('in_file', 'inputnode.in_file')]),
        # Feed IQMs computation
        (inputnode, iqmswf, [('in_file', 'inputnode.in_file'),
                             ('metadata', 'inputnode.metadata'),
                             ('entities', 'inputnode.entities')]),
        (hmcwf, iqmswf, [
            ('outputnode.out_fd', 'inputnode.hmc_fd'),
            ('outputnode.ref_frame', 'inputnode.ref_frame'),
        ]),
        # Feed reportlet generation
        (inputnode, pet_report_wf, [('in_file', 'inputnode.name_source')]),
        (hmcwf, pet_report_wf, [
            ('outputnode.out_fd', 'inputnode.hmc_fd'),
            ('outputnode.out_mot_param', 'inputnode.hmc_mot_param'),
        ]),
        (iqmswf, pet_report_wf, [
            ('outputnode.out_file', 'inputnode.in_iqms'),
        ]),
        (tacswf, pet_report_wf, [('outputnode.tacs_tsv', 'inputnode.tacs_tsv')]),
        (load_meta, pet_report_wf, [('out_dict', 'inputnode.metadata')]),
        (hmcwf, mean_pet, [('outputnode.out_file', 'in_file')]),
        (mean_pet, normwf, [('out_file', 'inputnode.pet_mean')]),
        (hmcwf, normwf, [('outputnode.out_file', 'inputnode.pet_dynamic')]),
        (normwf, tacswf, [('outputnode.pet_dynamic_t1', 'inputnode.pet_dynamic_t1')]),
        (load_meta, tacswf, [('out_dict', 'inputnode.pet_json')]),
        (hmcwf, outputnode, [('outputnode.out_fd', 'out_fd')]),
        (mean_pet, outputnode, [('out_file', 'pet_mean')]),
        (normwf, outputnode, [
            ('outputnode.pet_dseg', 'pet_dseg'),
            ('outputnode.out_report', 'norm_report')
        ]),
        (tacswf, outputnode, [('outputnode.tacs_tsv', 'tacs_tsv')]),
        (normwf, norm_report_sink, [('outputnode.out_report', 'in_file')]),
        (inputnode, norm_report_sink, [('in_file', 'source_file')]),
        (normwf, carpet_plot, [('outputnode.pet_dynamic_t1', 'in_pet'),
                           ('outputnode.pet_dseg', 'seg_file')]),
        (load_meta, carpet_plot, [('out_dict', 'metadata')]),
        (carpet_plot, ds_report_carpet, [('out_file', 'in_file')]),
        (inputnode, ds_report_carpet, [('in_file', 'source_file')]),
        (tacswf, ds_tacs, [('outputnode.tacs_tsv', 'in_file')]),
        (inputnode, ds_tacs, [('in_file', 'source_file')]),
    ])

    if not config.execution.no_sub:
        from mriqc.interfaces.webapi import UploadIQMs

        upldwf = pe.MapNode(
            UploadIQMs(
                endpoint=config.execution.webapi_url,
                auth_token=config.execution.webapi_token,
                strict=config.execution.upload_strict,
            ),
            name='UploadMetrics',
            iterfield=['in_iqms'],
        )

        workflow.connect([
            (iqmswf, upldwf, [('outputnode.out_file', 'in_iqms')]),
        ])

    return workflow


def hmc(name='petHMC', omp_nthreads=None):
    """
    Create a :abbr: petHMC (head motion correction) workflow.
    """
    from nipype.algorithms.confounds import FramewiseDisplacement
    from nipype.interfaces.afni import Volreg

    from mriqc.interfaces.pet import ChooseRefHMC

    mem_gb = config.workflow.biggest_file_gb['pet']

    workflow = pe.Workflow(name=name)

    inputnode = pe.Node(
        niu.IdentityInterface(fields=['in_file', 'fd_radius']),
        name='inputnode',
    )

    outputnode = pe.Node(
        niu.IdentityInterface(fields=['out_file', 'out_mot_param', 'out_fd', 'mpars', 'ref_frame']),
        name='outputnode',
    )

    choose_ref_node = pe.Node(
        ChooseRefHMC(),
        name='ChooseRefHMC',
    )

    estimate_hm = pe.Node(
        Volreg(args='-Fourier -twopass', zpad=4, outputtype='NIFTI_GZ'),
        name='estimate_hm',
        mem_gb=mem_gb * 2.5,
    )

    fdnode = pe.Node(
        FramewiseDisplacement(normalize=False, parameter_source='AFNI'),
        name='ComputeFD',
    )

    workflow.connect([
        (inputnode, choose_ref_node, [('in_file', 'in_file')]),
        (inputnode, estimate_hm, [('in_file', 'in_file')]),
        (inputnode, fdnode, [('fd_radius', 'radius')]),
        (choose_ref_node, estimate_hm, [('out_file', 'basefile')]),

        # Output corrected 4D PET file (out_file)
        (estimate_hm, outputnode, [
            ('out_file', 'out_file'),             # <-- added corrected 4D PET
            ('oned_file', 'out_mot_param'),
            ('oned_file', 'mpars'),
        ]),
        (estimate_hm, fdnode, [('oned_file', 'in_file')]),
        (fdnode, outputnode, [('out_file', 'out_fd')]),
        (choose_ref_node, outputnode, [('ref_frame', 'ref_frame')]),
    ])

    return workflow


def compute_iqms(name='ComputeIQMs'):
    """
    Initialize the workflow that actually computes the IQMs.

    .. workflow::

        from mriqc.workflows.functional.base import compute_iqms
        from mriqc.testing import mock_config
        with mock_config():
            wf = compute_iqms()

    """
    from nipype.interfaces.freesurfer import MRIConvert
    from nipype.interfaces.utility import Function

    from mriqc.interfaces import IQMFileSink
    from mriqc.interfaces.pet import FDStats
    from mriqc.interfaces.reports import AddProvenance


    mem_gb = config.workflow.biggest_file_gb['pet']

    workflow = pe.Workflow(name=name)
    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                'in_file',
                'metadata',
                'entities',
                'hmc_fd',
                'ref_frame',
                'fd_thres',
            ]
        ),
        name='inputnode',
    )
    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                'out_file',
                'fwhm_list',
                'fwhm_mean',
            ]
        ),
        name='outputnode',
    )

    # Set FD threshold
    inputnode.inputs.fd_thres = config.workflow.fd_thres

    # Compute FD statistics
    fd_stats = pe.Node(
        FDStats(),
        name='FDStats',
    )

    # Split 4D PET data into 3D frames using MRIConvert
    split_pet = pe.Node(
        MRIConvert(split=True, out_type='niigz'),
        name='SplitPET',
        mem_gb=mem_gb * 2,
    )

    # Compute smoothness (FWHM) per frame using MapNode
    fwhm_per_frame = pe.MapNode(
        Function(
            input_names=['in_file'],
            output_names=['fwhm_acf'],
            function=compute_acf_fwhm
        ),
        name='FWHMPerFrame',
        iterfield=['in_file']
    )

    mean_fwhm = pe.Node(
        niu.Function(input_names=['inlist'], output_names=['out'], function=_mean),
        name='FWHMMean'
    )

    addprov = pe.MapNode(
        AddProvenance(modality='pet'),
        name='provenance',
        run_without_submitting=True,
        iterfield=['in_file'],
    )

    # Save to JSON file
    datasink = pe.MapNode(
        IQMFileSink(
            modality='pet',
            out_dir=str(config.execution.output_dir),
            dataset=config.execution.dsname,
        ),
        name='datasink',
        run_without_submitting=True,
        iterfield=['in_file', 'root', 'metadata', 'provenance'],
    )

    # fmt: off
    workflow.connect([
        (inputnode, addprov, [('in_file', 'in_file')]),
        (inputnode, datasink, [('in_file', 'in_file'),
                            ('entities', 'entities'),
                            ('metadata', 'metadata'),
                            ('ref_frame', 'ref_frame')]),
        (inputnode, fd_stats, [('hmc_fd', 'in_fd'),
                            ('fd_thres', 'fd_thres')]),
        (inputnode, split_pet, [('in_file', 'in_file')]),
        (split_pet, fwhm_per_frame, [('out_file', 'in_file')]),
        (fwhm_per_frame, mean_fwhm, [('fwhm_acf', 'inlist')]),
        (mean_fwhm, datasink, [('out', 'fwhm_mean')]),
        (addprov, datasink, [('out_prov', 'provenance')]),
        (fd_stats, datasink, [('out_fd', 'root')]),
        (fwhm_per_frame, datasink, [('fwhm_acf', 'fwhm_per_frame')]),
        (datasink, outputnode, [('out_file', 'out_file')]),
        (fwhm_per_frame, outputnode, [('fwhm_acf', 'fwhm_list')]),
        (mean_fwhm, outputnode, [('out', 'fwhm_mean')])
    ])
    # fmt: on

    return workflow


def compute_acf_fwhm(in_file):
    """Return the ACF-based FWHM estimated by AFNI's 3dFWHMx."""
    import subprocess

    cmd = f'3dFWHMx -input {in_file} -combine -detrend -acf -automask'
    result = subprocess.run(cmd.split(), capture_output=True, text=True)

    output_lines = result.stdout.strip().split('\n')

    acf_line = None
    for line in output_lines:
        if line.startswith(" 0.") or line.startswith("0."):
            values = line.split()
            if len(values) >= 4:
                acf_line = values
                break

    if acf_line is None:
        raise ValueError('Failed to parse AFNI 3dFWHMx output correctly.')

    fwhm_acf = float(acf_line[3])

    return fwhm_acf


def _mean(inlist):
    from numpy import mean

    return float(mean(inlist))


def pet_mni_align(name='PETSpatialNormalization'):
    """Align the PET series to the SPM T1 template with corresponding Hammer's atlas"""
    import os.path as op

    from nipype.interfaces.utility import IdentityInterface
    from nipype.pipeline import engine as pe
    from pkg_resources import resource_filename

    workflow = pe.Workflow(name=name)

    inputnode = pe.Node(
        IdentityInterface(fields=['pet_mean', 'pet_dynamic']),
        name='inputnode',
    )

    outputnode = pe.Node(
        IdentityInterface(fields=['pet_dynamic_t1', 'pet_dseg', 'out_report']),
        name='outputnode',
    )

    template_dir = resource_filename('mriqc', 'data/atlas')
    template_t1 = op.join(template_dir, 'tpl-SPM_space-MNI152_desc-conform_T1.nii.gz')
    template_dseg = op.join(template_dir, 'tpl-SPM_space-MNI152_desc-conform_dseg.nii.gz')
    template_mask = op.join(template_dir, 'tpl-SPM_space-MNI152_desc-conform_mask.nii.gz')

    ants_norm = pe.Node(
        RobustMNINormalization(
            moving='boldref',
            reference='boldref',
            explicit_masking=True,
            float=True,
            generate_report=True,
            reference_image=template_t1,
            reference_mask=template_mask,
            settings=[op.join(resource_filename('mriqc', 'data/atlas'),
                              'petref-mni_registration_precise_000.json')],
        ),
        name='ANTsNormalization'
    )

    apply_transform_dynamic = pe.Node(
        ApplyTransforms(
            interpolation='Linear',
            input_image_type=3,  # Time-series
            dimension=3,
            float=True,
            reference_image=template_t1
        ),
        name='ApplyTransformDynamic'
    )

    workflow.connect([
        (inputnode, ants_norm, [('pet_mean', 'moving_image')]),

        # Correct connection: using composite transform
        (ants_norm, apply_transform_dynamic, [('composite_transform', 'transforms')]),

        # Connect dynamic PET
        (inputnode, apply_transform_dynamic, [('pet_dynamic', 'input_image')]),

        # Output
        (apply_transform_dynamic, outputnode, [('output_image', 'pet_dynamic_t1')]),

        (ants_norm, outputnode, [('out_report', 'out_report')])
    ])

    outputnode.inputs.pet_dseg = template_dseg

    return workflow


def extract_tacs(name='ExtractTACs'):
    """Extract time-activity curves from normalized dynamic PET."""
    import os.path as op

    from nipype.interfaces.utility import Function
    from nipype.pipeline import engine as pe

    workflow = pe.Workflow(name=name)

    inputnode = pe.Node(
        IdentityInterface(fields=['pet_dynamic_t1', 'pet_json']),
        name='inputnode'
    )

    outputnode = pe.Node(
        IdentityInterface(fields=['tacs_tsv']),
        name='outputnode'
    )

    template_dir = resource_filename('mriqc', 'data/atlas')
    labels_tsv = op.join(template_dir, 'tpl-SPM_space-MNI152_dseg.tsv')
    template_dseg = op.join(template_dir, 'tpl-SPM_space-MNI152_desc-conform_dseg.nii.gz')

    def compute_tacs(dseg_file, dynamic_pet, labels_tsv, pet_json):
        import os

        import nibabel as nib
        import numpy as np
        import pandas as pd

        dseg = nib.load(dseg_file).get_fdata()
        pet_data = nib.load(dynamic_pet).get_fdata()
        labels_df = pd.read_csv(labels_tsv, sep='\t')

        frame_times_start = np.array(pet_json['FrameTimesStart'])
        frame_duration = np.array(pet_json['FrameDuration'])
        frame_times_end = frame_times_start + frame_duration

        tacs = {
            'frame_times_start': frame_times_start,
            'frame_times_end': frame_times_end
        }

        for _, row in labels_df.iterrows():
            label_id = row['index']
            region_name = row['name']
            mask = dseg == label_id
            tac = pet_data[mask, :].mean(axis=0)
            tacs[region_name] = tac

        df = pd.DataFrame(tacs)
        tsv_file = os.path.abspath('tacs.tsv')
        df.to_csv(tsv_file, sep='\t', index=False)
        return tsv_file

    tac_extraction = pe.Node(
        Function(
            input_names=['dseg_file', 'dynamic_pet', 'labels_tsv', 'pet_json'],
            output_names=['tacs_tsv'],
            function=compute_tacs,
        ),
        name='TACExtraction'
    )

    tac_extraction.inputs.labels_tsv = labels_tsv
    tac_extraction.inputs.dseg_file = template_dseg

    workflow.connect([
        (inputnode, tac_extraction, [
            ('pet_dynamic_t1', 'dynamic_pet'),
            ('pet_json', 'pet_json')
        ]),
        (tac_extraction, outputnode, [('tacs_tsv', 'tacs_tsv')]),
    ])

    return workflow


def create_pet_carpet_plot(in_pet, seg_file, metadata, output_file):
    """Create a carpet plot grouped by tissue type."""

    import matplotlib.pyplot as plt
    import nibabel as nb
    import numpy as np
    import pandas as pd
    from pkg_resources import resource_filename

    pet_img = nb.load(in_pet)
    seg_img = nb.load(seg_file)
    seg_data = seg_img.get_fdata().astype(int)  # Extract segmentation data as numpy array

    template_dir = resource_filename('mriqc', 'data/atlas')
    labels_tsv = op.join(template_dir, 'tpl-SPM_space-MNI152_dseg.tsv')
    labels_df = pd.read_csv(labels_tsv, sep='\t')
    
    # Define labels based on segmentation values
    map_labels = {
        'Cortical': 1,
        'Subcortical': 2,
        'Cerebellar': 3,
    }

    cortical_keywords = [
        'gyrus',
        'cortex',
        'cingulate',
        'frontal',
        'temporal',
        'parietal',
        'occipital',
        'insula',
        'cuneus',
    ]
    subcortical_keywords = [
        'caudate',
        'putamen',
        'thalamus',
        'pallidum',
        'accumbens',
        'amygdala',
        'hippocampus',
    ]
    cerebellar_keywords = ['cerebellum']

    # Create a mapping from original labels to simplified labels
    label_to_group = {0: 0}  # Explicitly handle background
    for _, row in labels_df.iterrows():
        label_id = row['index']
        region_name = row['name'].lower()

        if any(keyword in region_name for keyword in cortical_keywords):
            label_to_group[label_id] = map_labels['Cortical']
        elif any(keyword in region_name for keyword in subcortical_keywords):
            label_to_group[label_id] = map_labels['Subcortical']
        elif any(keyword in region_name for keyword in cerebellar_keywords):
            label_to_group[label_id] = map_labels['Cerebellar']

    # Remap the segmentation image explicitly ensuring no gaps or unknown labels
    grouped_dseg_data = np.zeros_like(seg_data, dtype=int)
    unique_labels = np.unique(seg_data)

    for original_label in unique_labels:
        grouped_dseg_data[seg_data == original_label] = label_to_group.get(
            original_label, 0
        )  # Ensure fallback to 0

    # Ensure data has correct datatype for nilearn
    grouped_dseg_data = grouped_dseg_data.astype(np.int32)

    # Generate new segmentation NIfTI image
    grouped_dseg_img = nb.Nifti1Image(grouped_dseg_data, seg_img.affine, seg_img.header)

    fig, ax = plt.subplots(figsize=(15, 20))

    # Directly use plot_carpet and capture the figure object
    fig = plot_carpet(
        img=pet_img,
        mask_img=grouped_dseg_img,
        mask_labels=map_labels,
        t_r=None,
        cmap='turbo',
        cmap_labels='Set1',
        title='Global uptake patterns over time separated by tissue type'
    )

    # Manually adjust the xlabel on the correct axis (the carpet plot axis is usually axes[1])
    fig.axes[1].set_xlabel('Frame #', fontsize=14)

    # Adjust and save figure
    fig.tight_layout()
    output_file = op.abspath(output_file)
    fig.savefig(output_file, bbox_inches='tight', dpi=300)
    plt.close(fig)

    return output_file
