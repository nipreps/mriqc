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
"""Nipype interfaces to support anatomical workflow."""

from pathlib import Path

import nibabel as nb
import numpy as np
import scipy.ndimage as nd
from nipype.interfaces.base import (
    BaseInterfaceInputSpec,
    File,
    InputMultiPath,
    SimpleInterface,
    TraitedSpec,
    traits,
)
from nipype.utils.filemanip import fname_presuffix

from mriqc.qc.anatomical import (
    art_qi1,
    art_qi2,
    cjv,
    cnr,
    efc,
    fber,
    rpve,
    snr,
    snr_dietrich,
    summary_stats,
    volume_fraction,
    wm2max,
)
from mriqc.utils.misc import _flatten_dict


class StructuralQCInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True, desc='file to be plotted')
    in_noinu = File(exists=True, mandatory=True, desc='image after INU correction')
    in_segm = File(exists=True, mandatory=True, desc='segmentation file from FSL FAST')
    in_bias = File(exists=True, mandatory=True, desc='bias file')
    head_msk = File(exists=True, mandatory=True, desc='head mask')
    air_msk = File(exists=True, mandatory=True, desc='air mask')
    rot_msk = File(exists=True, mandatory=True, desc='rotation mask')
    artifact_msk = File(exists=True, mandatory=True, desc='air mask')
    in_pvms = InputMultiPath(
        File(exists=True),
        mandatory=True,
        desc='partial volume maps from FSL FAST',
    )
    in_tpms = InputMultiPath(File(), desc='tissue probability maps from FSL FAST')
    mni_tpms = InputMultiPath(File(), desc='tissue probability maps from FSL FAST')
    in_fwhm = traits.List(traits.Float, mandatory=True, desc='smoothness estimated with AFNI')
    human = traits.Bool(True, usedefault=True, desc='human workflow')


class StructuralQCOutputSpec(TraitedSpec):
    summary = traits.Dict(desc='summary statistics per tissue')
    icvs = traits.Dict(desc='intracranial volume (ICV) fractions')
    rpve = traits.Dict(desc='partial volume fractions')
    size = traits.Dict(desc='image sizes')
    spacing = traits.Dict(desc='image sizes')
    fwhm = traits.Dict(desc='full width half-maximum measure')
    inu = traits.Dict(desc='summary statistics of the bias field')
    snr = traits.Dict
    snrd = traits.Dict
    cnr = traits.Float
    fber = traits.Float
    efc = traits.Float
    qi_1 = traits.Float
    wm2max = traits.Float
    cjv = traits.Float
    out_qc = traits.Dict(desc='output flattened dictionary with all measures')
    out_noisefit = File(exists=True, desc='plot of background noise and chi fitting')
    tpm_overlap = traits.Dict


class StructuralQC(SimpleInterface):
    """
    Computes anatomical :abbr:`QC (Quality Control)` measures on the
    structural image given as input

    """

    input_spec = StructuralQCInputSpec
    output_spec = StructuralQCOutputSpec

    def _run_interface(self, runtime):  # pylint: disable=R0914,E1101
        imnii = nb.load(self.inputs.in_noinu)

        # Load image corrected for INU
        inudata = np.nan_to_num(imnii.get_fdata())
        inudata[inudata < 0] = 0

        if np.all(inudata < 1e-5):
            raise RuntimeError(
                'Input inhomogeneity-corrected data seem empty. '
                'MRIQC failed to process this dataset.'
            )

        # Load binary segmentation from FSL FAST
        segnii = nb.load(self.inputs.in_segm)
        segdata = np.asanyarray(segnii.dataobj).astype(np.uint8)

        if np.sum(segdata > 0) < 1e3:
            raise RuntimeError(
                'Input segmentation data is likely corrupt. MRIQC failed to process this dataset.'
            )

        # Load air, artifacts and head masks
        airdata = np.asanyarray(nb.load(self.inputs.air_msk).dataobj).astype(np.uint8)
        artdata = np.asanyarray(nb.load(self.inputs.artifact_msk).dataobj).astype(np.uint8)

        headdata = np.asanyarray(nb.load(self.inputs.head_msk).dataobj).astype(np.uint8)
        if np.sum(headdata > 0) < 100:
            raise RuntimeError(
                'Detected less than 100 voxels belonging to the head mask. '
                'MRIQC failed to process this dataset.'
            )

        rotdata = np.asanyarray(nb.load(self.inputs.rot_msk).dataobj).astype(np.uint8)

        # Load brain tissue probability maps from GMM segmentation
        pvms = {
            label: nb.load(fname).get_fdata()
            for label, fname in zip(('csf', 'gm', 'wm'), self.inputs.in_pvms)
        }
        pvmdata = list(pvms.values())

        # Add probability maps
        pvms['bg'] = airdata

        # Summary stats
        stats = summary_stats(inudata, pvms)
        self._results['summary'] = stats

        # SNR
        snrvals = []
        self._results['snr'] = {}
        for tlabel in ('csf', 'wm', 'gm'):
            snrvals.append(
                snr(
                    stats[tlabel]['median'],
                    stats[tlabel]['stdv'],
                    stats[tlabel]['n'],
                )
            )
            self._results['snr'][tlabel] = snrvals[-1]
        self._results['snr']['total'] = float(np.mean(snrvals))

        snrvals = []
        self._results['snrd'] = {
            tlabel: snr_dietrich(
                stats[tlabel]['median'],
                mad_air=stats['bg']['mad'],
                sigma_air=stats['bg']['stdv'],
            )
            for tlabel in ['csf', 'wm', 'gm']
        }
        self._results['snrd']['total'] = float(
            np.mean([val for _, val in list(self._results['snrd'].items())])
        )

        # CNR
        self._results['cnr'] = cnr(
            stats['wm']['median'],
            stats['gm']['median'],
            stats['bg']['stdv'],
            stats['wm']['stdv'],
            stats['gm']['stdv'],
        )

        # FBER
        self._results['fber'] = fber(inudata, headdata, rotdata)

        # EFC
        self._results['efc'] = efc(inudata, rotdata)

        # M2WM
        self._results['wm2max'] = wm2max(inudata, stats['wm']['median'])

        # Artifacts
        self._results['qi_1'] = art_qi1(airdata, artdata)

        # CJV
        self._results['cjv'] = cjv(
            # mu_wm, mu_gm, sigma_wm, sigma_gm
            stats['wm']['median'],
            stats['gm']['median'],
            stats['wm']['mad'],
            stats['gm']['mad'],
        )

        # FWHM
        fwhm = np.array(self.inputs.in_fwhm[:3]) / np.array(imnii.header.get_zooms()[:3])
        self._results['fwhm'] = {
            'x': float(fwhm[0]),
            'y': float(fwhm[1]),
            'z': float(fwhm[2]),
            'avg': float(np.average(fwhm)),
        }

        # ICVs
        self._results['icvs'] = volume_fraction(pvmdata)

        # RPVE
        self._results['rpve'] = rpve(pvmdata, segdata)

        # Image specs
        self._results['size'] = {
            'x': int(inudata.shape[0]),
            'y': int(inudata.shape[1]),
            'z': int(inudata.shape[2]),
        }
        self._results['spacing'] = {
            i: float(v) for i, v in zip(['x', 'y', 'z'], imnii.header.get_zooms()[:3])
        }

        try:
            self._results['size']['t'] = int(inudata.shape[3])
        except IndexError:
            pass

        try:
            self._results['spacing']['tr'] = float(imnii.header.get_zooms()[3])
        except IndexError:
            pass

        # Bias
        bias = nb.load(self.inputs.in_bias).get_fdata()[segdata > 0]
        self._results['inu'] = {
            'range': float(np.abs(np.percentile(bias, 95.0) - np.percentile(bias, 5.0))),
            'med': float(np.median(bias)),
        }  # pylint: disable=E1101

        mni_tpms = [nb.load(tpm).get_fdata() for tpm in self.inputs.mni_tpms]
        in_tpms = [nb.load(tpm).get_fdata() for tpm in self.inputs.in_pvms]
        overlap = fuzzy_jaccard(in_tpms, mni_tpms)
        self._results['tpm_overlap'] = {
            'csf': overlap[0],
            'gm': overlap[1],
            'wm': overlap[2],
        }

        # Flatten the dictionary
        self._results['out_qc'] = _flatten_dict(self._results)
        return runtime


class _ArtifactMaskInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True, desc='File to be plotted')
    head_mask = File(exists=True, mandatory=True, desc='head mask')
    glabella_xyz = traits.Tuple(
        (0.0, 90.0, -14.0),
        types=(traits.Float, traits.Float, traits.Float),
        usedefault=True,
        desc='position of the top of the glabella in standard coordinates',
    )
    inion_xyz = traits.Tuple(
        (0.0, -120.0, -14.0),
        types=(traits.Float, traits.Float, traits.Float),
        usedefault=True,
        desc='position of the top of the inion in standard coordinates',
    )
    ind2std_xfm = File(exists=True, mandatory=True, desc='individual to standard affine transform')
    zscore = traits.Float(10.0, usedefault=True, desc='z-score to consider artifacts')


class _ArtifactMaskOutputSpec(TraitedSpec):
    out_hat_msk = File(exists=True, desc='output "hat" mask')
    out_art_msk = File(exists=True, desc='output artifacts mask')
    out_air_msk = File(exists=True, desc='output "hat" mask, without artifacts')


class ArtifactMask(SimpleInterface):
    """
    Computes the artifact mask using the method described in [Mortamet2009]_.
    """

    input_spec = _ArtifactMaskInputSpec
    output_spec = _ArtifactMaskOutputSpec

    def _run_interface(self, runtime):
        from nibabel.affines import apply_affine
        from nitransforms.linear import Affine

        in_file = Path(self.inputs.in_file)
        imnii = nb.as_closest_canonical(nb.load(in_file))
        imdata = np.nan_to_num(imnii.get_fdata().astype(np.float32))

        xfm = Affine.from_filename(self.inputs.ind2std_xfm, fmt='itk')

        ras2ijk = np.linalg.inv(imnii.affine)
        glabella_ijk, inion_ijk = apply_affine(
            ras2ijk, xfm.map([self.inputs.glabella_xyz, self.inputs.inion_xyz])
        )

        hmdata = np.bool_(nb.load(self.inputs.head_mask).dataobj)

        # Calculate distance to border
        dist = nd.morphology.distance_transform_edt(~hmdata)

        hmdata[:, :, : int(inion_ijk[2])] = 1
        hmdata[:, (hmdata.shape[1] // 2) :, : int(glabella_ijk[2])] = 1

        dist[~hmdata] = 0
        dist /= dist.max()

        # Run the artifact detection
        qi1_img = artifact_mask(imdata, (~hmdata), dist, zscore=self.inputs.zscore)

        fname = in_file.relative_to(in_file.parent).stem
        ext = ''.join(in_file.suffixes)

        outdir = Path(runtime.cwd).absolute()
        self._results['out_hat_msk'] = str(outdir / f'{fname}_hat{ext}')
        self._results['out_art_msk'] = str(outdir / f'{fname}_art{ext}')
        self._results['out_air_msk'] = str(outdir / f'{fname}_air{ext}')

        hdr = imnii.header.copy()
        hdr.set_data_dtype(np.uint8)
        imnii.__class__(qi1_img.astype(np.uint8), imnii.affine, hdr).to_filename(
            self._results['out_art_msk']
        )

        airdata = (~hmdata).astype(np.uint8)
        imnii.__class__(airdata, imnii.affine, hdr).to_filename(self._results['out_hat_msk'])

        airdata[qi1_img > 0] = 0
        imnii.__class__(airdata.astype(np.uint8), imnii.affine, hdr).to_filename(
            self._results['out_air_msk']
        )
        return runtime


class ComputeQI2InputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True, desc='File to be plotted')
    air_msk = File(exists=True, mandatory=True, desc='air (without artifacts) mask')


class ComputeQI2OutputSpec(TraitedSpec):
    qi2 = traits.Float(desc='computed QI2 value')
    out_file = File(desc='output plot: noise fit')


class ComputeQI2(SimpleInterface):
    """
    Computes the artifact mask using the method described in [Mortamet2009]_.
    """

    input_spec = ComputeQI2InputSpec
    output_spec = ComputeQI2OutputSpec

    def _run_interface(self, runtime):
        imdata = nb.load(self.inputs.in_file).get_fdata()
        airdata = nb.load(self.inputs.air_msk).get_fdata()
        qi2, out_file = art_qi2(imdata, airdata)
        self._results['qi2'] = qi2
        self._results['out_file'] = out_file
        return runtime


class HarmonizeInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True, desc='input data (after bias correction)')
    wm_mask = File(exists=True, mandatory=True, desc='white-matter mask')
    erodemsk = traits.Bool(True, usedefault=True, desc='erode mask')
    thresh = traits.Float(0.9, usedefault=True, desc='WM probability threshold')


class HarmonizeOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='input data (after intensity harmonization)')


class Harmonize(SimpleInterface):
    """
    Computes the artifact mask using the method described in [Mortamet2009]_.
    """

    input_spec = HarmonizeInputSpec
    output_spec = HarmonizeOutputSpec

    def _run_interface(self, runtime):
        in_file = nb.load(self.inputs.in_file)
        wm_mask = nb.load(self.inputs.wm_mask).get_fdata()
        wm_mask[wm_mask < 0.9] = 0
        wm_mask[wm_mask > 0] = 1
        wm_mask = wm_mask.astype(np.uint8)

        if self.inputs.erodemsk:
            # Create a structural element to be used in an opening operation.
            struct = nd.generate_binary_structure(3, 2)
            # Perform an opening operation on the background data.
            wm_mask = nd.binary_erosion(wm_mask, structure=struct).astype(np.uint8)

        data = in_file.get_fdata()
        data *= 1000.0 / np.median(data[wm_mask > 0])

        out_file = fname_presuffix(self.inputs.in_file, suffix='_harmonized', newpath='.')
        in_file.__class__(data, in_file.affine, in_file.header).to_filename(out_file)

        self._results['out_file'] = out_file

        return runtime


class RotationMaskInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True, desc='input data')


class RotationMaskOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='rotation mask (if any)')


class RotationMask(SimpleInterface):
    """
    Computes the artifact mask using the method described in [Mortamet2009]_.
    """

    input_spec = RotationMaskInputSpec
    output_spec = RotationMaskOutputSpec

    def _run_interface(self, runtime):
        in_file = nb.load(self.inputs.in_file)
        data = in_file.get_fdata()
        mask = data <= 0

        # Pad one pixel to control behavior on borders of binary_opening
        mask = np.pad(mask, pad_width=(1,), mode='constant', constant_values=1)

        # Remove noise
        struct = nd.generate_binary_structure(3, 2)
        mask = nd.binary_opening(mask, structure=struct).astype(np.uint8)

        # Remove small objects
        label_im, nb_labels = nd.label(mask)
        if nb_labels > 2:
            sizes = nd.sum(mask, label_im, list(range(nb_labels + 1)))
            ordered = sorted(zip(sizes, list(range(nb_labels + 1))), reverse=True)
            for _, label in ordered[2:]:
                mask[label_im == label] = 0

        # Un-pad
        mask = mask[1:-1, 1:-1, 1:-1]

        # If mask is small, clean-up
        if mask.sum() < 500:
            mask = np.zeros_like(mask, dtype=np.uint8)

        out_img = in_file.__class__(mask, in_file.affine, in_file.header)
        out_img.header.set_data_dtype(np.uint8)

        out_file = fname_presuffix(self.inputs.in_file, suffix='_rotmask', newpath='.')
        out_img.to_filename(out_file)
        self._results['out_file'] = out_file
        return runtime


def artifact_mask(imdata, airdata, distance, zscore=10.0):
    """Compute a mask of artifacts found in the air region."""
    from statsmodels.robust.scale import mad

    qi1_msk = np.zeros(imdata.shape, dtype=bool)
    bg_data = imdata[airdata]
    if (bg_data > 0).sum() < 10:
        return qi1_msk

    # Standardize the distribution of the background
    bg_spread = mad(bg_data[bg_data > 0])
    bg_data[bg_data > 0] = bg_data[bg_data > 0] / bg_spread

    # Apply this threshold to the background voxels to identify voxels
    # contributing artifacts.
    qi1_msk[airdata] = bg_data > zscore
    qi1_msk[distance < 0.10] = False

    # Create a structural element to be used in an opening operation.
    struct = nd.generate_binary_structure(3, 1)
    qi1_msk = nd.binary_opening(qi1_msk, struct).astype(np.uint8)
    return qi1_msk


def fuzzy_jaccard(in_tpms, in_mni_tpms):
    overlaps = []
    for tpm, mni_tpm in zip(in_tpms, in_mni_tpms):
        tpm = tpm.reshape(-1)
        mni_tpm = mni_tpm.reshape(-1)

        num = np.min([tpm, mni_tpm], axis=0).sum()
        den = np.max([tpm, mni_tpm], axis=0).sum()
        overlaps.append(float(num / den))
    return overlaps
