# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

from os import path as op

from pkg_resources import resource_filename as pkgrf
import numpy as np
import nibabel as nb

from nipype.interfaces.base import (
    traits,
    TraitedSpec,
    BaseInterfaceInputSpec,
    File,
    isdefined,
    SimpleInterface,
)
from nipype.interfaces.ants import ApplyTransforms
from .. import config


class ConformImageInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True, desc="input image")
    check_ras = traits.Bool(True, usedefault=True, desc="check that orientation is RAS")
    check_dtype = traits.Bool(True, usedefault=True, desc="check data type")


class ConformImageOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc="output conformed file")


class ConformImage(SimpleInterface):

    """
    Conforms an input image

    List of nifti datatypes:

    .. note: original Analyze 7.5 types

      DT_NONE                    0
      DT_UNKNOWN                 0     / what it says, dude           /
      DT_BINARY                  1     / binary (1 bit/voxel)         /
      DT_UNSIGNED_CHAR           2     / unsigned char (8 bits/voxel) /
      DT_SIGNED_SHORT            4     / signed short (16 bits/voxel) /
      DT_SIGNED_INT              8     / signed int (32 bits/voxel)   /
      DT_FLOAT                  16     / float (32 bits/voxel)        /
      DT_COMPLEX                32     / complex (64 bits/voxel)      /
      DT_DOUBLE                 64     / double (64 bits/voxel)       /
      DT_RGB                   128     / RGB triple (24 bits/voxel)   /
      DT_ALL                   255     / not very useful (?)          /

    .. note: added names for the same data types

      DT_UINT8                   2
      DT_INT16                   4
      DT_INT32                   8
      DT_FLOAT32                16
      DT_COMPLEX64              32
      DT_FLOAT64                64
      DT_RGB24                 128


    .. note: new codes for NIFTI

      DT_INT8                  256     / signed char (8 bits)         /
      DT_UINT16                512     / unsigned short (16 bits)     /
      DT_UINT32                768     / unsigned int (32 bits)       /
      DT_INT64                1024     / long long (64 bits)          /
      DT_UINT64               1280     / unsigned long long (64 bits) /
      DT_FLOAT128             1536     / long double (128 bits)       /
      DT_COMPLEX128           1792     / double pair (128 bits)       /
      DT_COMPLEX256           2048     / long double pair (256 bits)  /
      NIFTI_TYPE_UINT8           2 /! unsigned char. /
      NIFTI_TYPE_INT16           4 /! signed short. /
      NIFTI_TYPE_INT32           8 /! signed int. /
      NIFTI_TYPE_FLOAT32        16 /! 32 bit float. /
      NIFTI_TYPE_COMPLEX64      32 /! 64 bit complex = 2 32 bit floats. /
      NIFTI_TYPE_FLOAT64        64 /! 64 bit float = double. /
      NIFTI_TYPE_RGB24         128 /! 3 8 bit bytes. /
      NIFTI_TYPE_INT8          256 /! signed char. /
      NIFTI_TYPE_UINT16        512 /! unsigned short. /
      NIFTI_TYPE_UINT32        768 /! unsigned int. /
      NIFTI_TYPE_INT64        1024 /! signed long long. /
      NIFTI_TYPE_UINT64       1280 /! unsigned long long. /
      NIFTI_TYPE_FLOAT128     1536 /! 128 bit float = long double. /
      NIFTI_TYPE_COMPLEX128   1792 /! 128 bit complex = 2 64 bit floats. /
      NIFTI_TYPE_COMPLEX256   2048 /! 256 bit complex = 2 128 bit floats /

    """

    input_spec = ConformImageInputSpec
    output_spec = ConformImageOutputSpec

    def _run_interface(self, runtime):
        # Squeeze 4th dimension if possible (#660)
        nii = nb.squeeze_image(nb.load(self.inputs.in_file))
        hdr = nii.header.copy()
        if self.inputs.check_ras:
            nii = nb.as_closest_canonical(nii)

        if self.inputs.check_dtype:
            changed = True
            datatype = int(hdr["datatype"])

            if datatype == 1:
                config.loggers.interface.warning(
                    'Input image %s has a suspicious data type "%s"',
                    self.inputs.in_file,
                    hdr.get_data_dtype(),
                )

            # signed char and bool to uint8
            if datatype == 1 or datatype == 2 or datatype == 256:
                dtype = np.uint8

            # int16 to uint16
            elif datatype == 4:
                dtype = np.uint16

            # Signed long, long long, etc to uint32
            elif datatype == 8 or datatype == 1024 or datatype == 1280:
                dtype = np.uint32

            # Floats over 32 bits
            elif datatype == 64 or datatype == 1536:
                dtype = np.float32
            else:
                changed = False

            if changed:
                hdr.set_data_dtype(dtype)
                nii = nb.Nifti1Image(nii.get_data().astype(dtype), nii.affine, hdr)

        # Generate name
        out_file, ext = op.splitext(op.basename(self.inputs.in_file))
        if ext == ".gz":
            out_file, ext2 = op.splitext(out_file)
            ext = ext2 + ext

        self._results["out_file"] = op.abspath("{}_conformed{}".format(out_file, ext))
        nii.to_filename(self._results["out_file"])
        return runtime


class EnsureSizeInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, copyfile=False, mandatory=True, desc="input image")
    in_mask = File(exists=True, copyfile=False, desc="input mask")
    pixel_size = traits.Float(2.0, usedefault=True, desc="desired pixel size (mm)")


class EnsureSizeOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc="output image")
    out_mask = File(exists=True, desc="output mask")


class EnsureSize(SimpleInterface):
    """
    Checks the size of the input image and resamples it to
    have `pixel_size`

    """

    input_spec = EnsureSizeInputSpec
    output_spec = EnsureSizeOutputSpec

    def _run_interface(self, runtime):
        nii = nb.load(self.inputs.in_file)
        zooms = nii.header.get_zooms()
        size_diff = np.array(zooms[:3]) - (self.inputs.pixel_size - 0.1)
        if np.all(size_diff >= -1e-3):
            config.loggers.interface.info("Voxel size is large enough")
            self._results["out_file"] = self.inputs.in_file
            if isdefined(self.inputs.in_mask):
                self._results["out_mask"] = self.inputs.in_mask
            return runtime

        config.loggers.interface.info(
            "One or more voxel dimensions (%f, %f, %f) are smaller than "
            "the requested voxel size (%f) - diff=(%f, %f, %f)",
            zooms[0],
            zooms[1],
            zooms[2],
            self.inputs.pixel_size,
            size_diff[0],
            size_diff[1],
            size_diff[2],
        )

        # Figure out new matrix
        # 1) Get base affine
        aff_base = nii.header.get_base_affine()
        aff_base_inv = np.linalg.inv(aff_base)

        # 2) Find center pixel in mm
        center_idx = (np.array(nii.shape[:3]) - 1) * 0.5
        center_mm = aff_base.dot(center_idx.tolist() + [1])

        # 3) Find extent of each dimension
        min_mm = aff_base.dot([-0.5, -0.5, -0.5, 1])
        max_mm = aff_base.dot((np.array(nii.shape[:3]) - 0.5).tolist() + [1])
        extent_mm = np.abs(max_mm - min_mm)[:3]

        # 4) Find new matrix size
        new_size = np.array(extent_mm / self.inputs.pixel_size, dtype=int)

        # 5) Initialize new base affine
        new_base = (
            aff_base[:3, :3] * np.abs(aff_base_inv[:3, :3]) * self.inputs.pixel_size
        )

        # 6) Find new center
        new_center_idx = (new_size - 1) * 0.5
        new_affine_base = np.eye(4)
        new_affine_base[:3, :3] = new_base
        new_affine_base[:3, 3] = center_mm[:3] - new_base.dot(new_center_idx)

        # 7) Rotate new matrix
        rotation = nii.affine.dot(aff_base_inv)
        new_affine = rotation.dot(new_affine_base)

        # 8) Generate new reference image
        hdr = nii.header.copy()
        hdr.set_data_shape(new_size)
        ref_file = "resample_ref.nii.gz"
        nb.Nifti1Image(
            np.zeros(new_size, dtype=nii.get_data_dtype()), new_affine, hdr
        ).to_filename(ref_file)

        out_prefix, ext = op.splitext(op.basename(self.inputs.in_file))
        if ext == ".gz":
            out_prefix, ext2 = op.splitext(out_prefix)
            ext = ext2 + ext

        out_file = op.abspath("%s_resampled%s" % (out_prefix, ext))

        # 9) Resample new image
        ApplyTransforms(
            dimension=3,
            input_image=self.inputs.in_file,
            reference_image=ref_file,
            interpolation="LanczosWindowedSinc",
            transforms=[pkgrf("mriqc", "data/itk_identity.tfm")],
            output_image=out_file,
        ).run()

        self._results["out_file"] = out_file

        if isdefined(self.inputs.in_mask):
            hdr = nii.header.copy()
            hdr.set_data_shape(new_size)
            hdr.set_data_dtype(np.uint8)
            ref_mask = "mask_ref.nii.gz"
            nb.Nifti1Image(
                np.zeros(new_size, dtype=np.uint8), new_affine, hdr
            ).to_filename(ref_mask)

            out_mask = op.abspath("%s_resmask%s" % (out_prefix, ext))
            ApplyTransforms(
                dimension=3,
                input_image=self.inputs.in_mask,
                reference_image=ref_mask,
                interpolation="NearestNeighbor",
                transforms=[pkgrf("mriqc", "data/itk_identity.tfm")],
                output_image=out_mask,
            ).run()

            self._results["out_mask"] = out_mask

        return runtime
