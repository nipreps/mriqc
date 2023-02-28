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
"""
Anatomical workflow
===================

.. image :: _static/anatomical_workflow_source.svg

The anatomical workflow follows the following steps:

#. Conform (reorientations, revise data types) input data and read
   associated metadata.
#. Skull-stripping (AFNI).
#. Calculate head mask -- :py:func:`headmsk_wf`.
#. Spatial Normalization to MNI (ANTs)
#. Calculate air mask above the nasial-cerebelum plane -- :py:func:`airmsk_wf`.
#. Brain tissue segmentation (FAST).
#. Extraction of IQMs -- :py:func:`compute_iqms`.
#. Individual-reports generation -- :py:func:`individual_reports`.

This workflow is orchestrated by :py:func:`anat_qc_workflow`.

For the skull-stripping, we use ``afni_wf`` from ``niworkflows.anat.skullstrip``:

.. workflow::

    from niworkflows.anat.skullstrip import afni_wf
    from mriqc.testing import mock_config
    with mock_config():
        wf = afni_wf()

"""
from mriqc import config
from mriqc.interfaces import (
    ArtifactMask,
    ComputeQI2,
    ConformImage,
    IQMFileSink,
    RotationMask,
    StructuralQC,
)
from mriqc.interfaces.reports import AddProvenance
from mriqc.interfaces.datalad import DataladIdentityInterface
from mriqc.messages import BUILDING_WORKFLOW
from mriqc.workflows.utils import get_fwhmx
from nipype.interfaces import ants, fsl
from nipype.interfaces import io as nio
from nipype.interfaces import utility as niu
from nipype.pipeline import engine as pe
from templateflow.api import get as get_template


def anat_qc_workflow(name="anatMRIQC"):
    """
    One-subject-one-session-one-run pipeline to extract the NR-IQMs from
    anatomical images

    .. workflow::

        import os.path as op
        from mriqc.workflows.anatomical import anat_qc_workflow
        from mriqc.testing import mock_config
        with mock_config():
            wf = anat_qc_workflow()

    """

    dataset = config.workflow.inputs.get("T1w", []) + config.workflow.inputs.get(
        "T2w", []
    )

    message = BUILDING_WORKFLOW.format(
        detail=(
            f"for {len(dataset)} NIfTI files." if len(dataset) > 2
            else f"({' and '.join(('<%s>' % v for v in dataset))})."
        ),
    )
    config.loggers.workflow.info(message)

    # Initialize workflow
    workflow = pe.Workflow(name=name)

    # Define workflow, inputs and outputs
    # 0. Get data
    inputnode = pe.Node(niu.IdentityInterface(fields=["in_file"]), name="inputnode")
    inputnode.iterables = [("in_file", dataset)]

    datalad_get = pe.Node(DataladIdentityInterface(
        fields=["in_file"],
        dataset_path=config.execution.bids_dir
    ), name="datalad_get")

    outputnode = pe.Node(niu.IdentityInterface(fields=["out_json"]), name="outputnode")

    # 1. Reorient anatomical image
    to_ras = pe.Node(ConformImage(check_dtype=False), name="conform")
    # 2. species specific skull-stripping
    if config.workflow.species.lower() == "human":
        skull_stripping = synthstrip_wf(omp_nthreads=config.nipype.omp_nthreads)
        ss_bias_field = "outputnode.bias_image"
    else:
        from nirodents.workflows.brainextraction import init_rodent_brain_extraction_wf

        skull_stripping = init_rodent_brain_extraction_wf(
            template_id=config.workflow.template_id
        )
        ss_bias_field = "final_n4.bias_image"
    # 3. Head mask
    hmsk = headmsk_wf()
    # 4. Spatial Normalization, using ANTs
    norm = spatial_normalization()
    # 5. Air mask (with and without artifacts)
    amw = airmsk_wf()
    # 6. Brain tissue segmentation
    if config.workflow.species.lower() == "human":
        segment = pe.Node(
            fsl.FAST(segments=True, out_basename="segment"),
            name="segmentation",
            mem_gb=5,
        )
        seg_in_file = "in_files"
        dseg_out = "tissue_class_map"
        pve_out = "partial_volume_files"
    else:
        from niworkflows.interfaces.fixes import ApplyTransforms

        tpms = [
            str(tpm)
            for tpm in get_template(
                config.workflow.template_id, label=["CSF", "GM", "WM"], suffix="probseg"
            )
        ]

        xfm_tpms = pe.MapNode(
            ApplyTransforms(
                dimension=3,
                default_value=0,
                float=True,
                interpolation="Gaussian",
                output_image="prior.nii.gz",
            ),
            iterfield=["input_image"],
            name="xfm_tpms",
        )
        xfm_tpms.inputs.input_image = tpms

        format_tpm_names = pe.Node(
            niu.Function(
                input_names=["in_files"],
                output_names=["file_format"],
                function=_format_tpm_names,
                execution={"keep_inputs": True, "remove_unnecessary_outputs": False},
            ),
            name="format_tpm_names",
        )

        segment = pe.Node(
            ants.Atropos(
                initialization="PriorProbabilityImages",
                number_of_tissue_classes=3,
                prior_weighting=0.1,
                mrf_radius=[1, 1, 1],
                mrf_smoothing_factor=0.01,
                save_posteriors=True,
                out_classified_image_name="segment.nii.gz",
                output_posteriors_name_template="segment_%02d.nii.gz",
            ),
            name="segmentation",
            mem_gb=5,
        )
        seg_in_file = "intensity_images"
        dseg_out = "classified_image"
        pve_out = "posteriors"

    # 7. Compute IQMs
    iqmswf = compute_iqms()
    # Reports
    repwf = individual_reports()

    # Connect all nodes
    # fmt: off
    workflow.connect([
        (inputnode, datalad_get, [("in_file", "in_file")]),
        (datalad_get, to_ras, [("in_file", "in_file")]),
        (datalad_get, iqmswf, [("in_file", "inputnode.in_file")]),
        (datalad_get, norm, [(("in_file", _get_mod), "inputnode.modality")]),
        (to_ras, skull_stripping, [("out_file", "inputnode.in_files")]),
        (skull_stripping, segment, [("outputnode.out_brain", seg_in_file)]),
        (skull_stripping, hmsk, [("outputnode.out_corrected", "inputnode.in_file")]),
        (segment, hmsk, [(dseg_out, "inputnode.in_segm")]),
        (skull_stripping, norm, [
            ("outputnode.out_corrected", "inputnode.moving_image"),
            ("outputnode.out_mask", "inputnode.moving_mask")]),
        (norm, amw, [
            ("outputnode.inverse_composite_transform", "inputnode.inverse_composite_transform")]),
        (norm, iqmswf, [
            ("outputnode.inverse_composite_transform", "inputnode.inverse_composite_transform")]),
        (norm, repwf, ([
            ("outputnode.out_report", "inputnode.mni_report")])),
        (to_ras, amw, [("out_file", "inputnode.in_file")]),
        (skull_stripping, amw, [("outputnode.out_mask", "inputnode.in_mask")]),
        (hmsk, amw, [("outputnode.out_file", "inputnode.head_mask")]),
        (to_ras, iqmswf, [("out_file", "inputnode.in_ras")]),
        (skull_stripping, iqmswf, [("outputnode.out_corrected", "inputnode.inu_corrected"),
                                   (ss_bias_field, "inputnode.in_inu"),
                                   ("outputnode.out_mask", "inputnode.brainmask")]),
        (amw, iqmswf, [("outputnode.air_mask", "inputnode.airmask"),
                       ("outputnode.hat_mask", "inputnode.hatmask"),
                       ("outputnode.art_mask", "inputnode.artmask"),
                       ("outputnode.rot_mask", "inputnode.rotmask")]),
        (segment, iqmswf, [(dseg_out, "inputnode.segmentation"),
                           (pve_out, "inputnode.pvms")]),
        (hmsk, iqmswf, [("outputnode.out_file", "inputnode.headmask")]),
        (to_ras, repwf, [("out_file", "inputnode.in_ras")]),
        (skull_stripping, repwf, [
            ("outputnode.out_corrected", "inputnode.inu_corrected"),
            ("outputnode.out_mask", "inputnode.brainmask")]),
        (hmsk, repwf, [("outputnode.out_file", "inputnode.headmask")]),
        (amw, repwf, [("outputnode.air_mask", "inputnode.airmask"),
                      ("outputnode.art_mask", "inputnode.artmask"),
                      ("outputnode.rot_mask", "inputnode.rotmask")]),
        (segment, repwf, [(dseg_out, "inputnode.segmentation")]),
        (iqmswf, repwf, [("outputnode.noisefit", "inputnode.noisefit")]),
        (iqmswf, repwf, [("outputnode.out_file", "inputnode.in_iqms")]),
        (iqmswf, outputnode, [("outputnode.out_file", "out_json")]),
    ])

    if config.workflow.species.lower() == 'human':
        workflow.connect([
            (datalad_get, segment, [(("in_file", _get_imgtype), "img_type")]),
        ])
    else:
        workflow.connect([
            (skull_stripping, xfm_tpms, [("outputnode.out_brain", "reference_image")]),
            (norm, xfm_tpms, [("outputnode.inverse_composite_transform", "transforms")]),
            (xfm_tpms, format_tpm_names, [('output_image', 'in_files')]),
            (format_tpm_names, segment, [(('file_format', _pop), 'prior_image')]),
            (skull_stripping, segment, [("outputnode.out_mask", "mask_image")]),
        ])
    # fmt: on

    # Upload metrics
    if not config.execution.no_sub:
        from ..interfaces.webapi import UploadIQMs

        upldwf = pe.Node(UploadIQMs(), name="UploadMetrics")
        upldwf.inputs.url = config.execution.webapi_url
        upldwf.inputs.strict = config.execution.upload_strict
        if config.execution.webapi_port:
            upldwf.inputs.port = config.execution.webapi_port

        # fmt: off
        workflow.connect([
            (iqmswf, upldwf, [("outputnode.out_file", "in_iqms")]),
            (upldwf, repwf, [("api_id", "inputnode.api_id")]),
        ])
        # fmt: on

    return workflow


def spatial_normalization(name="SpatialNormalization"):
    """Create a simplied workflow to perform fast spatial normalization."""
    from niworkflows.interfaces.reportlets.registration import (
        SpatialNormalizationRPT as RobustMNINormalization,
    )

    # Have the template id handy
    tpl_id = config.workflow.template_id

    # Define workflow interface
    workflow = pe.Workflow(name=name)
    inputnode = pe.Node(
        niu.IdentityInterface(fields=["moving_image", "moving_mask", "modality"]),
        name="inputnode",
    )
    outputnode = pe.Node(
        niu.IdentityInterface(fields=["inverse_composite_transform", "out_report"]),
        name="outputnode",
    )

    # Spatial normalization
    norm = pe.Node(
        RobustMNINormalization(
            flavor=["testing", "fast"][config.execution.debug],
            num_threads=config.nipype.omp_nthreads,
            float=config.execution.ants_float,
            template=tpl_id,
            generate_report=True,
        ),
        name="SpatialNormalization",
        # Request all MultiProc processes when ants_nthreads > n_procs
        num_threads=config.nipype.omp_nthreads,
        mem_gb=3,
    )
    if config.workflow.species.lower() == "human":
        norm.inputs.reference_mask = str(
            get_template(tpl_id, resolution=2, desc="brain", suffix="mask")
        )
    else:
        norm.inputs.reference_image = str(get_template(tpl_id, suffix="T2w"))
        norm.inputs.reference_mask = str(
            get_template(tpl_id, desc="brain", suffix="mask")[0]
        )

    # fmt: off
    workflow.connect([
        (inputnode, norm, [("moving_image", "moving_image"),
                           ("moving_mask", "moving_mask"),
                           ("modality", "reference")]),
        (norm, outputnode, [("inverse_composite_transform", "inverse_composite_transform"),
                            ("out_report", "out_report")]),
    ])
    # fmt: on

    return workflow


def compute_iqms(name="ComputeIQMs"):
    """
    Setup the workflow that actually computes the IQMs.

    .. workflow::

        from mriqc.workflows.anatomical import compute_iqms
        from mriqc.testing import mock_config
        with mock_config():
            wf = compute_iqms()

    """
    from niworkflows.interfaces.bids import ReadSidecarJSON

    from ..interfaces.anatomical import Harmonize
    from .utils import _tofloat

    workflow = pe.Workflow(name=name)
    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                "in_file",
                "in_ras",
                "brainmask",
                "airmask",
                "artmask",
                "headmask",
                "rotmask",
                "hatmask",
                "segmentation",
                "inu_corrected",
                "in_inu",
                "pvms",
                "metadata",
                "inverse_composite_transform",
            ]
        ),
        name="inputnode",
    )
    outputnode = pe.Node(
        niu.IdentityInterface(fields=["out_file", "noisefit"]),
        name="outputnode",
    )

    # Extract metadata
    meta = pe.Node(ReadSidecarJSON(), name="metadata")

    # Add provenance
    addprov = pe.Node(AddProvenance(), name="provenance", run_without_submitting=True)

    # AFNI check smoothing
    fwhm_interface = get_fwhmx()

    fwhm = pe.Node(fwhm_interface, name="smoothness")

    # Harmonize
    homog = pe.Node(Harmonize(), name="harmonize")
    if config.workflow.species.lower() != "human":
        homog.inputs.erodemsk = False
        homog.inputs.thresh = 0.8

    # Mortamet's QI2
    getqi2 = pe.Node(ComputeQI2(), name="ComputeQI2")

    # Compute python-coded measures
    measures = pe.Node(
        StructuralQC(human=config.workflow.species.lower() == "human"), "measures"
    )

    # Project MNI segmentation to T1 space
    invt = pe.MapNode(
        ants.ApplyTransforms(
            dimension=3, default_value=0, interpolation="Linear", float=True
        ),
        iterfield=["input_image"],
        name="MNItpms2t1",
    )
    if config.workflow.species.lower() == "human":
        invt.inputs.input_image = [
            str(p)
            for p in get_template(
                config.workflow.template_id,
                suffix="probseg",
                resolution=1,
                label=["CSF", "GM", "WM"],
            )
        ]
    else:
        invt.inputs.input_image = [
            str(p)
            for p in get_template(
                config.workflow.template_id,
                suffix="probseg",
                label=["CSF", "GM", "WM"],
            )
        ]

    datasink = pe.Node(
        IQMFileSink(
            out_dir=config.execution.output_dir,
            dataset=config.execution.dsname,
        ),
        name="datasink",
        run_without_submitting=True,
    )

    def _getwm(inlist):
        return inlist[-1]

    # fmt: off
    workflow.connect([
        (inputnode, meta, [("in_file", "in_file")]),
        (inputnode, datasink, [("in_file", "in_file"),
                               (("in_file", _get_mod), "modality")]),
        (inputnode, addprov, [(("in_file", _get_mod), "modality")]),
        (meta, datasink, [("subject", "subject_id"),
                          ("session", "session_id"),
                          ("task", "task_id"),
                          ("acquisition", "acq_id"),
                          ("reconstruction", "rec_id"),
                          ("run", "run_id"),
                          ("out_dict", "metadata")]),
        (inputnode, addprov, [("in_file", "in_file"),
                              ("airmask", "air_msk"),
                              ("rotmask", "rot_msk")]),
        (inputnode, getqi2, [("in_ras", "in_file"),
                             ("hatmask", "air_msk")]),
        (inputnode, homog, [("inu_corrected", "in_file"),
                            (("pvms", _getwm), "wm_mask")]),
        (inputnode, measures, [("in_inu", "in_bias"),
                               ("in_ras", "in_file"),
                               ("airmask", "air_msk"),
                               ("headmask", "head_msk"),
                               ("artmask", "artifact_msk"),
                               ("rotmask", "rot_msk"),
                               ("segmentation", "in_segm"),
                               ("pvms", "in_pvms")]),
        (inputnode, fwhm, [("in_ras", "in_file"),
                           ("brainmask", "mask")]),
        (inputnode, invt, [("in_ras", "reference_image"),
                           ("inverse_composite_transform", "transforms")]),
        (homog, measures, [("out_file", "in_noinu")]),
        (invt, measures, [("output_image", "mni_tpms")]),
        (fwhm, measures, [(("fwhm", _tofloat), "in_fwhm")]),
        (measures, datasink, [("out_qc", "root")]),
        (addprov, datasink, [("out_prov", "provenance")]),
        (getqi2, datasink, [("qi2", "qi_2")]),
        (getqi2, outputnode, [("out_file", "noisefit")]),
        (datasink, outputnode, [("out_file", "out_file")]),
    ])
    # fmt: on

    return workflow


def individual_reports(name="ReportsWorkflow"):
    """
    Generate the components of the individual report.

    .. workflow::

        from mriqc.workflows.anatomical import individual_reports
        from mriqc.testing import mock_config
        with mock_config():
            wf = individual_reports()

    """
    from nireports.interfaces.viz import PlotMosaic
    from ..interfaces.reports import IndividualReport

    verbose = config.execution.verbose_reports
    pages = 2
    extra_pages = int(verbose) * 7

    workflow = pe.Workflow(name=name)
    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                "in_ras",
                "brainmask",
                "headmask",
                "airmask",
                "artmask",
                "rotmask",
                "segmentation",
                "inu_corrected",
                "noisefit",
                "in_iqms",
                "mni_report",
                "api_id",
            ]
        ),
        name="inputnode",
    )

    mosaic_zoom = pe.Node(
        PlotMosaic(out_file="plot_anat_mosaic1_zoomed.svg", cmap="Greys_r"),
        name="PlotMosaicZoomed",
    )

    mosaic_noise = pe.Node(
        PlotMosaic(
            out_file="plot_anat_mosaic2_noise.svg",
            only_noise=True,
            cmap="viridis_r",
        ),
        name="PlotMosaicNoise",
    )

    mplots = pe.Node(niu.Merge(pages + extra_pages), name="MergePlots")
    rnode = pe.Node(IndividualReport(), name="GenerateReport")

    # Link images that should be reported
    dsplots = pe.Node(
        nio.DataSink(
            base_directory=str(config.execution.output_dir),
            parameterization=False,
        ),
        name="dsplots",
        run_without_submitting=True,
    )

    # fmt: off
    workflow.connect([
        (inputnode, rnode, [("in_iqms", "in_iqms")]),
        (inputnode, mosaic_zoom, [("in_ras", "in_file"),
                                  ("brainmask", "bbox_mask_file")]),
        (inputnode, mosaic_noise, [("in_ras", "in_file")]),
        (mosaic_zoom, mplots, [("out_file", "in1")]),
        (mosaic_noise, mplots, [("out_file", "in2")]),
        (mplots, rnode, [("out", "in_plots")]),
        (rnode, dsplots, [("out_file", "@html_report")]),
    ])
    # fmt: on

    if not verbose:
        return workflow

    from nireports.interfaces.viz import PlotContours

    plot_segm = pe.Node(
        PlotContours(
            display_mode="z",
            levels=[0.5, 1.5, 2.5],
            cut_coords=10,
            colors=["r", "g", "b"],
        ),
        name="PlotSegmentation",
    )

    plot_bmask = pe.Node(
        PlotContours(
            display_mode="z",
            levels=[0.5],
            colors=["r"],
            cut_coords=10,
            out_file="bmask",
        ),
        name="PlotBrainmask",
    )
    plot_airmask = pe.Node(
        PlotContours(
            display_mode="x",
            levels=[0.5],
            colors=["r"],
            cut_coords=6,
            out_file="airmask",
        ),
        name="PlotAirmask",
    )
    plot_headmask = pe.Node(
        PlotContours(
            display_mode="x",
            levels=[0.5],
            colors=["r"],
            cut_coords=6,
            out_file="headmask",
        ),
        name="PlotHeadmask",
    )
    plot_artmask = pe.Node(
        PlotContours(
            display_mode="z",
            levels=[0.5],
            colors=["r"],
            cut_coords=10,
            out_file="artmask",
            saturate=True,
        ),
        name="PlotArtmask",
    )

    # fmt: off
    workflow.connect([
        (inputnode, plot_segm, [("in_ras", "in_file"),
                                ("segmentation", "in_contours")]),
        (inputnode, plot_bmask, [("in_ras", "in_file"),
                                 ("brainmask", "in_contours")]),
        (inputnode, plot_headmask, [("in_ras", "in_file"),
                                    ("headmask", "in_contours")]),
        (inputnode, plot_airmask, [("in_ras", "in_file"),
                                   ("airmask", "in_contours")]),
        (inputnode, plot_artmask, [("in_ras", "in_file"),
                                   ("artmask", "in_contours")]),
        (inputnode, mplots, [("mni_report", f"in{pages + 1}")]),
        (plot_bmask, mplots, [("out_file", f"in{pages + 2}")]),
        (plot_segm, mplots, [("out_file", f"in{pages + 3}")]),
        (plot_artmask, mplots, [("out_file", f"in{pages + 4}")]),
        (plot_headmask, mplots, [("out_file", f"in{pages + 5}")]),
        (plot_airmask, mplots, [("out_file", f"in{pages + 6}")]),
        (inputnode, mplots, [("noisefit", f"in{pages + 7}")]),
    ])
    # fmt: on

    return workflow


def headmsk_wf(name="HeadMaskWorkflow"):
    """
    Computes a head mask as in [Mortamet2009]_.

    .. workflow::

        from mriqc.testing import mock_config
        from mriqc.workflows.anatomical import headmsk_wf
        with mock_config():
            wf = headmsk_wf()

    """

    use_bet = config.workflow.headmask.upper() == "BET"
    has_dipy = False

    if not use_bet:
        try:
            from dipy.denoise import nlmeans  # noqa

            has_dipy = True
        except ImportError:
            pass

    if not use_bet and not has_dipy:
        raise RuntimeError(
            "DIPY is not installed and ``config.workflow.headmask`` is not BET."
        )

    workflow = pe.Workflow(name=name)
    inputnode = pe.Node(
        niu.IdentityInterface(fields=["in_file", "in_segm"]), name="inputnode"
    )
    outputnode = pe.Node(niu.IdentityInterface(fields=["out_file"]), name="outputnode")

    if use_bet:
        # Alternative for when dipy is not installed
        bet = pe.Node(fsl.BET(surfaces=True), name="fsl_bet")

        # fmt: off
        workflow.connect([
            (inputnode, bet, [("in_file", "in_file")]),
            (bet, outputnode, [('outskin_mask_file', "out_file")]),
        ])
        # fmt: on

    else:
        from nipype.interfaces.dipy import Denoise

        enhance = pe.Node(
            niu.Function(
                input_names=["in_file"],
                output_names=["out_file"],
                function=_enhance,
            ),
            name="Enhance",
        )
        estsnr = pe.Node(
            niu.Function(
                input_names=["in_file", "seg_file"],
                output_names=["out_snr"],
                function=_estimate_snr,
            ),
            name="EstimateSNR",
        )
        denoise = pe.Node(Denoise(), name="Denoise")
        gradient = pe.Node(
            niu.Function(
                input_names=["in_file", "snr", "sigma"],
                output_names=["out_file"],
                function=image_gradient,
            ),
            name="Grad",
        )
        thresh = pe.Node(
            niu.Function(
                input_names=["in_file", "in_segm", "aniso", "thresh"],
                output_names=["out_file"],
                function=gradient_threshold,
            ),
            name="GradientThreshold",
        )
        if config.workflow.species != "human":
            calc_sigma = pe.Node(
                niu.Function(
                    input_names=["in_file"],
                    output_names=["sigma"],
                    function=sigma_calc,
                ),
                name="calc_sigma",
            )
            workflow.connect(
                [
                    (inputnode, calc_sigma, [("in_file", "in_file")]),
                    (calc_sigma, gradient, [("sigma", "sigma")]),
                ]
            )

            thresh.inputs.aniso = True
            thresh.inputs.thresh = 4.0

        # fmt: off
        workflow.connect([
            (inputnode, estsnr, [("in_file", "in_file"),
                                 ("in_segm", "seg_file")]),
            (estsnr, denoise, [("out_snr", "snr")]),
            (inputnode, enhance, [("in_file", "in_file")]),
            (enhance, denoise, [("out_file", "in_file")]),
            (estsnr, gradient, [("out_snr", "snr")]),
            (denoise, gradient, [("out_file", "in_file")]),
            (inputnode, thresh, [("in_segm", "in_segm")]),
            (gradient, thresh, [("out_file", "in_file")]),
            (thresh, outputnode, [("out_file", "out_file")]),
        ])
        # fmt: on

    return workflow


def airmsk_wf(name="AirMaskWorkflow"):
    """
    Implements the Step 1 of [Mortamet2009]_.

    .. workflow::

        from mriqc.testing import mock_config
        from mriqc.workflows.anatomical import airmsk_wf
        with mock_config():
            wf = airmsk_wf()

    """
    workflow = pe.Workflow(name=name)

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                "in_file",
                "in_mask",
                "head_mask",
                "inverse_composite_transform",
            ]
        ),
        name="inputnode",
    )
    outputnode = pe.Node(
        niu.IdentityInterface(fields=["hat_mask", "air_mask", "art_mask", "rot_mask"]),
        name="outputnode",
    )

    rotmsk = pe.Node(RotationMask(), name="RotationMask")

    invt = pe.Node(
        ants.ApplyTransforms(
            dimension=3,
            default_value=0,
            interpolation="MultiLabel",
            float=True,
        ),
        name="invert_xfm",
    )
    if config.workflow.species.lower() == "human":
        invt.inputs.input_image = str(
            get_template(
                config.workflow.template_id, resolution=1, desc="head", suffix="mask"
            )
        )
    else:
        # TODO: provide options for other populations
        invt.inputs.input_image = str(
            get_template(config.workflow.template_id, desc="brain", suffix="mask")[0]
        )

    qi1 = pe.Node(ArtifactMask(), name="ArtifactMask")

    # fmt: off
    workflow.connect([
        (inputnode, rotmsk, [("in_file", "in_file")]),
        (inputnode, qi1, [("in_file", "in_file"),
                          ("head_mask", "head_mask")]),
        (rotmsk, qi1, [("out_file", "rot_mask")]),
        (inputnode, invt, [("in_mask", "reference_image"),
                           ("inverse_composite_transform", "transforms")]),
        (invt, qi1, [("output_image", "nasion_post_mask")]),
        (qi1, outputnode, [("out_hat_msk", "hat_mask"),
                           ("out_air_msk", "air_mask"),
                           ("out_art_msk", "art_mask")]),
        (rotmsk, outputnode, [("out_file", "rot_mask")])
    ])
    # fmt: on

    return workflow


def synthstrip_wf(name="synthstrip_wf", omp_nthreads=None):
    """Create a brain-extraction workflow using SynthStrip."""
    from nipype.interfaces.ants import N4BiasFieldCorrection
    from niworkflows.interfaces.nibabel import IntensityClip, ApplyMask
    from mriqc.interfaces.synthstrip import SynthStrip

    inputnode = pe.Node(niu.IdentityInterface(fields=["in_files"]), name="inputnode")
    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=["out_corrected", "out_brain", "bias_image", "out_mask"]
        ),
        name="outputnode",
    )

    # truncate target intensity for N4 correction
    pre_clip = pe.Node(IntensityClip(p_min=10, p_max=99.9), name="pre_clip")

    pre_n4 = pe.Node(
        N4BiasFieldCorrection(
            dimension=3,
            num_threads=omp_nthreads,
            rescale_intensities=True,
            copy_header=True,
        ),
        name="pre_n4",
    )

    post_n4 = pe.Node(
        N4BiasFieldCorrection(
            dimension=3,
            save_bias=True,
            num_threads=omp_nthreads,
            n_iterations=[50] * 4,
            copy_header=True,
        ),
        name="post_n4",
    )

    synthstrip = pe.Node(
        SynthStrip(),
        name="synthstrip",
    )

    final_masked = pe.Node(ApplyMask(), name="final_masked")
    final_inu = pe.Node(niu.Function(function=_apply_bias_correction), name="final_inu")

    workflow = pe.Workflow(name=name)
    # fmt: off
    workflow.connect([
        (inputnode, final_inu, [("in_files", "in_file")]),
        (inputnode, pre_clip, [("in_files", "in_file")]),
        (pre_clip, pre_n4, [("out_file", "input_image")]),
        (pre_n4, synthstrip, [("output_image", "in_file")]),
        (synthstrip, post_n4, [("out_mask", "weight_image")]),
        (synthstrip, final_masked, [("out_mask", "in_mask")]),
        (pre_clip, post_n4, [("out_file", "input_image")]),
        (post_n4, final_inu, [("bias_image", "bias_image")]),
        (post_n4, final_masked, [("output_image", "in_file")]),
        (final_masked, outputnode, [("out_file", "out_brain")]),
        (post_n4, outputnode, [("bias_image", "bias_image")]),
        (synthstrip, outputnode, [("out_mask", "out_mask")]),
        (post_n4, outputnode, [("output_image", "out_corrected")]),
    ])
    # fmt: on
    return workflow


def _apply_bias_correction(in_file, bias_image, out_file=None):
    import os.path as op

    import numpy as np
    import nibabel as nb

    img = nb.load(in_file)
    data = np.clip(
        img.get_fdata() * nb.load(bias_image).get_fdata(),
        a_min=0,
        a_max=None,
    )
    out_img = img.__class__(
        data.astype(img.get_data_dtype()),
        img.affine,
        img.header,
    )

    if out_file is None:
        fname, ext = op.splitext(op.basename(in_file))
        if ext == ".gz":
            fname, ext2 = op.splitext(fname)
            ext = ext2 + ext
        out_file = op.abspath("{}_inu{}".format(fname, ext))

    out_img.to_filename(out_file)
    return out_file


def _binarize(in_file, threshold=0.5, out_file=None):
    import os.path as op

    import nibabel as nb
    import numpy as np

    if out_file is None:
        fname, ext = op.splitext(op.basename(in_file))
        if ext == ".gz":
            fname, ext2 = op.splitext(fname)
            ext = ext2 + ext
        out_file = op.abspath("{}_bin{}".format(fname, ext))

    nii = nb.load(in_file)
    data = nii.get_data()

    data[data <= threshold] = 0
    data[data > 0] = 1

    hdr = nii.header.copy()
    hdr.set_data_dtype(np.uint8)
    nb.Nifti1Image(data.astype(np.uint8), nii.affine, hdr).to_filename(out_file)
    return out_file


def _estimate_snr(in_file, seg_file):
    import nibabel as nb
    import numpy as np
    from mriqc.qc.anatomical import snr

    data = nb.load(in_file).get_data()
    mask = nb.load(seg_file).get_data() == 2  # WM label
    out_snr = snr(np.mean(data[mask]), data[mask].std(), mask.sum())
    return out_snr


def _enhance(in_file, out_file=None):
    import os.path as op

    import nibabel as nb
    import numpy as np

    if out_file is None:
        fname, ext = op.splitext(op.basename(in_file))
        if ext == ".gz":
            fname, ext2 = op.splitext(fname)
            ext = ext2 + ext
        out_file = op.abspath(f"{fname}_enhanced{ext}")

    imnii = nb.load(in_file)
    data = imnii.get_data().astype(np.float32)  # pylint: disable=no-member
    range_max = np.percentile(data[data > 0], 99.98)
    range_min = np.median(data[data > 0])

    # Resample signal excess pixels
    excess = np.where(data > range_max)
    data[excess] = 0
    data[excess] = np.random.choice(data[data > range_min], size=len(excess[0]))

    nb.Nifti1Image(data, imnii.affine, imnii.header).to_filename(out_file)

    return out_file


def sigma_calc(in_file):
    import nibabel as nb

    zooms = nb.load(in_file).header.get_zooms()
    sigma = [(zoom / min(zooms)) * 3 for zoom in zooms]

    return sigma


def image_gradient(in_file, snr, sigma=3.0, out_file=None):
    """Computes the magnitude gradient of an image using numpy"""
    import os.path as op

    import nibabel as nb
    import numpy as np
    from scipy.ndimage import gaussian_gradient_magnitude as gradient

    if out_file is None:
        fname, ext = op.splitext(op.basename(in_file))
        if ext == ".gz":
            fname, ext2 = op.splitext(fname)
            ext = ext2 + ext
        out_file = op.abspath(f"{fname}_grad{ext}")

    imnii = nb.load(in_file)
    data = imnii.get_data().astype(np.float32)  # pylint: disable=no-member
    datamax = np.percentile(data.reshape(-1), 99.5)
    data *= 100 / datamax
    grad = gradient(data, sigma)
    gradmax = np.percentile(grad.reshape(-1), 99.5)
    grad *= 100.0
    grad /= gradmax

    nb.Nifti1Image(grad, imnii.affine, imnii.header).to_filename(out_file)
    return out_file


def gradient_threshold(in_file, in_segm, thresh=15.0, out_file=None, aniso=False):
    """Compute a threshold from the histogram of the magnitude gradient image"""
    import os.path as op

    import nibabel as nb
    import numpy as np
    from scipy import ndimage as sim

    if not aniso:
        struct = sim.iterate_structure(sim.generate_binary_structure(3, 2), 2)
    else:
        # Generate an anisotropic binary structure, taking into account slice thickness
        img = nb.load(in_file)
        zooms = img.header.get_zooms()
        dist = max(zooms)
        dim = img.header["dim"][0]

        x = np.ones((5) * np.ones(dim, dtype=np.int8))
        np.put(x, x.size // 2, 0)
        dist_matrix = np.round(sim.distance_transform_edt(x, sampling=zooms), 5)
        struct = dist_matrix <= dist

    if out_file is None:
        fname, ext = op.splitext(op.basename(in_file))
        if ext == ".gz":
            fname, ext2 = op.splitext(fname)
            ext = ext2 + ext
        out_file = op.abspath(f"{fname}_gradmask{ext}")

    imnii = nb.load(in_file)

    hdr = imnii.header.copy()
    hdr.set_data_dtype(np.uint8)  # pylint: disable=no-member

    data = imnii.get_data().astype(np.float32)

    mask = np.zeros_like(data, dtype=np.uint8)  # pylint: disable=no-member
    mask[data > thresh] = 1

    segdata = nb.load(in_segm).get_data().astype(np.uint8)
    segdata[segdata > 0] = 1
    segdata = sim.binary_dilation(segdata, struct, iterations=2, border_value=1).astype(
        np.uint8
    )
    mask[segdata > 0] = 1
    mask = sim.binary_closing(mask, struct, iterations=2).astype(np.uint8)
    # Remove small objects
    label_im, nb_labels = sim.label(mask)
    artmsk = np.zeros_like(mask)
    if nb_labels > 2:
        sizes = sim.sum(mask, label_im, list(range(nb_labels + 1)))
        ordered = list(reversed(sorted(zip(sizes, list(range(nb_labels + 1))))))
        for _, label in ordered[2:]:
            mask[label_im == label] = 0
            artmsk[label_im == label] = 1

    mask = sim.binary_fill_holes(mask, struct).astype(
        np.uint8
    )  # pylint: disable=no-member

    nb.Nifti1Image(mask, imnii.affine, hdr).to_filename(out_file)
    return out_file


def _get_imgtype(in_file):
    from pathlib import Path

    return int(Path(in_file).name.rstrip(".gz").rstrip(".nii").split("_")[-1][1])


def _get_mod(in_file):
    from pathlib import Path

    return Path(in_file).name.rstrip(".gz").rstrip(".nii").split("_")[-1]


def _format_tpm_names(in_files, fname_string=None):
    from pathlib import Path
    import nibabel as nb
    import glob

    out_path = Path.cwd().absolute()

    # copy files to cwd and rename iteratively
    for count, fname in enumerate(in_files):
        img = nb.load(fname)
        extension = "".join(Path(fname).suffixes)
        out_fname = f"priors_{1 + count:02}{extension}"
        nb.save(img, Path(out_path, out_fname))

    if fname_string is None:
        fname_string = f"priors_%02d{extension}"

    out_files = [
        str(prior) for prior in glob.glob(str(Path(out_path, f"priors*{extension}")))
    ]

    # return path with c-style format string for Atropos
    file_format = str(Path(out_path, fname_string))
    return file_format, out_files


def _pop(inlist):
    if isinstance(inlist, (list, tuple)):
        return inlist[0]
    return inlist
