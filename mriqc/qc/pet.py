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
import numpy as np
from pathlib import Path
import os.path as op
import matplotlib.pyplot as plt
from nipype.interfaces.base import SimpleInterface, BaseInterfaceInputSpec, TraitedSpec, File, traits, isdefined


class _PlotFDInputSpec(BaseInterfaceInputSpec):
    in_fd = File(
        exists=True,
        mandatory=True,
        desc='motion parameters for FD computation',
    )
    in_file = File(exists=True, mandatory=True, desc="File to be plotted")
    out_file = traits.File(exists=False, desc="output file name")


class _PlotFDOutputSpec(TraitedSpec):
    out_file = File(desc="Output file")


class PlotFD(SimpleInterface):
    input_spec = _PlotFDInputSpec
    output_spec = _PlotFDOutputSpec

    def _run_interface(self, runtime):
        import matplotlib.pyplot as plt
        import numpy as np
        import os

        # Load FD data from file
        fd_values = np.loadtxt(self.inputs.in_fd, skiprows=1)

        plt.figure(figsize=(12, 5))
        plt.plot(np.arange(len(fd_values)), fd_values, '-r')
        plt.xlabel('Frame number')
        plt.ylabel('Framewise Displacement (FD)')
        plt.title('FD plot for PET QC')
        plt.grid(True)

        output_filename = os.path.abspath('fd_plot.png')
        plt.savefig(output_filename, bbox_inches='tight')
        plt.close()

        self._results['out_file'] = output_filename
        return runtime


class _PlotRotationInputSpec(BaseInterfaceInputSpec):
    mot_param = File(exists=True, mandatory=True, desc="motion parameters")
    in_file = File(exists=True, mandatory=True, desc="File to be plotted")
    out_file = traits.File(exists=False, desc="output file name")


class _PlotRotationOutputSpec(TraitedSpec):
    out_file = File(desc="Output file")


class PlotRotation(SimpleInterface):
    input_spec = _PlotRotationInputSpec
    output_spec = _PlotRotationOutputSpec

    def _run_interface(self, runtime):
        #Define filename to save the plot
        in_file_ref = Path(self.inputs.in_file)
        if isdefined(self.inputs.out_file):
            in_file_ref = Path(self.inputs.out_file)

        fname = in_file_ref.name.rstrip("".join(in_file_ref.suffixes))
        out_file = (Path(runtime.cwd) / (f"plot_{fname}_rotations.png")).resolve()
        self._results["out_file"] = str(out_file)

        # Extract timeseries
        motion = np.loadtxt(self.inputs.mot_param)
        rot_angles = motion[:, 0:3]
        n_frames = rot_angles.shape[0]

        plt.figure(figsize=(11, 5))
        plt.plot(np.arange(0, n_frames), rot_angles[:, 0], '-r', label='rot_x')
        plt.plot(np.arange(0, n_frames), rot_angles[:, 1], '-g', label='rot_y')
        plt.plot(np.arange(0, n_frames), rot_angles[:, 2], '-b', label='rot_z')
        plt.legend(loc='upper left')
        plt.ylabel('Rotation [degrees]')
        plt.xlabel('frame #')
        plt.grid(visible=True)
        plt.savefig(out_file, format='png')
        plt.close()

        return runtime


class _PlotTranslationInputSpec(BaseInterfaceInputSpec):
    mot_param = File(exists=True, mandatory=True, desc="motion parameters")
    in_file = File(exists=True, mandatory=True, desc="File to be plotted")
    out_file = traits.File(exists=False, desc="output file name")


class _PlotTranslationOutputSpec(TraitedSpec):
    out_file = File(desc="Output file")


class PlotTranslation(SimpleInterface):
    input_spec = _PlotTranslationInputSpec
    output_spec = _PlotTranslationOutputSpec

    def _run_interface(self, runtime):
        # Define filename to save the plot
        in_file_ref = Path(self.inputs.in_file)
        if isdefined(self.inputs.out_file):
            in_file_ref = Path(self.inputs.out_file)

        fname = in_file_ref.name.rstrip("".join(in_file_ref.suffixes))
        out_file = (Path(runtime.cwd) / (f"plot_{fname}_translations.png")).resolve()
        self._results["out_file"] = str(out_file)

        # Extract timeseries
        motion = np.loadtxt(self.inputs.mot_param)
        translations = motion[:, 3:6]
        n_frames = translations.shape[0]

        plt.figure(figsize=(11, 5))
        plt.plot(np.arange(0, n_frames), translations[:, 0], '-r', label='trans_x')
        plt.plot(np.arange(0, n_frames), translations[:, 1], '-g', label='trans_y')
        plt.plot(np.arange(0, n_frames), translations[:, 2], '-b', label='trans_z')
        plt.legend(loc='upper left')
        plt.ylabel('Translation [mm]')
        plt.xlabel('frame #')
        plt.grid(visible=True)
        plt.savefig(out_file, format='png')
        plt.close()

        return runtime
