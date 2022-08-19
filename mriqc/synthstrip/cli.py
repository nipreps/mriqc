# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# Copyright 2022 The NiPreps Developers <nipreps@gmail.com>
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
# STATEMENT OF CHANGES: This file is derived from sources licensed under the FreeSurfer 1.0 license
# terms, and this file has been changed.
# The full licensing terms of the original work are found at:
# https://github.com/freesurfer/freesurfer/blob/2995ded957961a7f3704de57eee88eb6cc30d52d/LICENSE.txt
# A copy of the license has been archived in the ORIGINAL_LICENSE file
# found within this redistribution.
#
# The original file this work derives from is found at:
# https://github.com/freesurfer/freesurfer/blob/2995ded957961a7f3704de57eee88eb6cc30d52d/mri_synthstrip/mri_synthstrip
#
# [April 2022] CHANGES:
#    * MAINT: Split the monolithic file into model and CLI submodules
#    * ENH: Replace freesurfer Python bundle with in-house code.
#
"""
Robust, universal skull-stripping for brain images of any type.
If you use SynthStrip in your analysis, please cite:

  A Hoopes, JS Mora, AV Dalca, B Fischl, M Hoffmann.
  SynthStrip: Skull-Stripping for Any Brain Image.
  https://arxiv.org/abs/2203.09974

"""


def main():
    """Entry point to SynthStrip."""
    import os
    from argparse import ArgumentParser
    import numpy as np
    import scipy
    import nibabel as nb
    import torch
    from .model import StripModel

    # parse command line
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "-i",
        "--image",
        metavar="file",
        required=True,
        help="Input image to skullstrip.",
    )
    parser.add_argument(
        "-o", "--out", metavar="file", help="Save stripped image to path."
    )
    parser.add_argument(
        "-m", "--mask", metavar="file", help="Save binary brain mask to path."
    )
    parser.add_argument("-g", "--gpu", action="store_true", help="Use the GPU.")
    parser.add_argument(
        "-b",
        "--border",
        default=1,
        type=int,
        help="Mask border threshold in mm. Default is 1.",
    )
    parser.add_argument("--model", metavar="file", help="Alternative model weights.")
    args = parser.parse_args()

    # sanity check on the inputs
    if not args.out and not args.mask:
        parser.fatal("Must provide at least --out or --mask output flags.")

    # necessary for speed gains (I think)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    # configure GPU device
    if args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        device = torch.device("cuda")
        device_name = "GPU"
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        device = torch.device("cpu")
        device_name = "CPU"

    # configure model
    print(f"Configuring model on the {device_name}")

    with torch.no_grad():
        model = StripModel()
        model.to(device)
        model.eval()

    # load model weights
    if args.model is not None:
        modelfile = args.model
        print("Using custom model weights")
    else:
        raise RuntimeError("A model must be provided.")

    checkpoint = torch.load(modelfile, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    # load input volume
    print(f"Input image read from: {args.image}")

    # normalize intensities
    image = nb.load(args.image)
    conformed = conform(image)
    in_data = conformed.get_fdata(dtype="float32")
    in_data -= in_data.min()
    in_data = np.clip(in_data / np.percentile(in_data, 99), 0, 1)
    in_data = in_data[np.newaxis, np.newaxis]

    # predict the surface distance transform
    input_tensor = torch.from_numpy(in_data).to(device)
    with torch.no_grad():
        sdt = model(input_tensor).cpu().numpy().squeeze()

    # unconform the sdt and extract mask
    sdt_target = resample_like(
        nb.Nifti1Image(sdt, conformed.affine, None),
        image,
        output_dtype="int16",
        cval=100,
    )
    sdt_data = np.asanyarray(sdt_target.dataobj).astype("int16")

    # find largest CC (just do this to be safe for now)
    components = scipy.ndimage.label(sdt_data.squeeze() < args.border)[0]
    bincount = np.bincount(components.flatten())[1:]
    mask = components == (np.argmax(bincount) + 1)
    mask = scipy.ndimage.morphology.binary_fill_holes(mask)

    # write the masked output
    if args.out:
        img_data = image.get_fdata()
        bg = np.min([0, img_data.min()])
        img_data[mask == 0] = bg
        nb.Nifti1Image(img_data, image.affine, image.header).to_filename(
            args.out,
        )
        print(f"Masked image saved to: {args.out}")

    # write the brain mask
    if args.mask:
        hdr = image.header.copy()
        hdr.set_data_dtype("uint8")
        nb.Nifti1Image(mask, image.affine, hdr).to_filename(args.mask)
        print(f"Binary brain mask saved to: {args.mask}")

    print("If you use SynthStrip in your analysis, please cite:")
    print("----------------------------------------------------")
    print("SynthStrip: Skull-Stripping for Any Brain Image.")
    print("A Hoopes, JS Mora, AV Dalca, B Fischl, M Hoffmann.")


def conform(input_nii):
    """Resample image as SynthStrip likes it."""
    import numpy as np
    import nibabel as nb
    from nitransforms.linear import Affine

    shape = np.array(input_nii.shape[:3])
    affine = input_nii.affine

    # Get corner voxel centers in index coords
    corner_centers_ijk = (
        np.array(
            [
                (i, j, k)
                for k in (0, shape[2] - 1)
                for j in (0, shape[1] - 1)
                for i in (0, shape[0] - 1)
            ]
        )
        + 0.5
    )

    # Get corner voxel centers in mm
    corners_xyz = (
        affine
        @ np.hstack((corner_centers_ijk, np.ones((len(corner_centers_ijk), 1)))).T
    )

    # Target affine is 1mm voxels in LIA orientation
    target_affine = np.diag([-1.0, 1.0, -1.0, 1.0])[:, (0, 2, 1, 3)]

    # Target shape
    extent = corners_xyz.min(1)[:3], corners_xyz.max(1)[:3]
    target_shape = ((extent[1] - extent[0]) / 1.0 + 0.999).astype(int)

    # SynthStrip likes dimensions be multiple of 64 (192, 256, or 320)
    target_shape = np.clip(
        np.ceil(np.array(target_shape) / 64).astype(int) * 64, 192, 320
    )

    # Ensure shape ordering is LIA too
    target_shape[2], target_shape[1] = target_shape[1:3]

    # Coordinates of center voxel do not change
    input_c = affine @ np.hstack((0.5 * (shape - 1), 1.0))
    target_c = target_affine @ np.hstack((0.5 * (target_shape - 1), 1.0))

    # Rebase the origin of the new, plumb affine
    target_affine[:3, 3] -= target_c[:3] - input_c[:3]

    nii = Affine(
        reference=nb.Nifti1Image(np.zeros(target_shape), target_affine, None),
    ).apply(input_nii)
    return nii


def resample_like(image, target, output_dtype=None, cval=0):
    """Resample the input image to be in the target's grid via identity transform."""
    from nitransforms.linear import Affine

    return Affine(reference=target).apply(image, output_dtype=output_dtype, cval=cval)


if __name__ == "__main__":
    main()
