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
"""SynthStrip interface."""
import os
from pathlib import Path
from nipype.interfaces.base import (
    CommandLine,
    CommandLineInputSpec,
    File,
    TraitedSpec,
    traits,
    Undefined,
)

_fs_home = os.getenv("FREESURFER_HOME", None)
_default_model_path = Path(_fs_home) / "models" / "synthstrip.1.pt" if _fs_home else Undefined

if _fs_home and not _default_model_path.exists():
    _default_model_path = Undefined


class _SynthStripInputSpec(CommandLineInputSpec):
    in_file = File(
        exists=True,
        mandatory=True,
        argstr="-i %s",
        desc="Input image to be brain extracted",
    )
    use_gpu = traits.Bool(
        False, usedefault=True, argstr="-g", desc="Use GPU", nohash=True
    )
    model = File(
        str(_default_model_path),
        usedefault=True,
        exists=True,
        argstr="--model %s",
        desc="file containing model's weights",
    )
    border_mm = traits.Int(
        1, usedefault=True, argstr="-b %d", desc="Mask border threshold in mm"
    )
    out_file = File(
        name_source=["in_file"],
        name_template="%s_desc-brain.nii.gz",
        argstr="-o %s",
        desc="store brain-extracted input to file",
    )
    out_mask = File(
        name_source=["in_file"],
        name_template="%s_desc-brain_mask.nii.gz",
        argstr="-m %s",
        desc="store brainmask to file",
    )


class _SynthStripOutputSpec(TraitedSpec):
    out_file = File(desc="brain-extracted image")
    out_mask = File(desc="brain mask")


class SynthStrip(CommandLine):
    _cmd = "synthstrip"
    input_spec = _SynthStripInputSpec
    output_spec = _SynthStripOutputSpec
