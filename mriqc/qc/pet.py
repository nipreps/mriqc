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
import os
import pandas as pd
import seaborn as sns
import re


def setup_plot_style():
    sns.set_theme(style='whitegrid', font_scale=1.4, context='talk')
    plt.figure(figsize=(16, 10))


class _PlotFDInputSpec(BaseInterfaceInputSpec):
    in_fd = File(
        exists=True,
        mandatory=True,
        desc='motion parameters for FD computation',
    )
    in_file = File(exists=True, mandatory=True, desc="File to be plotted")
    metadata = traits.Dict(mandatory=True, desc='Metadata dictionary containing timing info')
    out_file = traits.File(exists=False, desc="output file name")


class _PlotFDOutputSpec(TraitedSpec):
    out_file = File(desc="Output file")


class PlotFD(SimpleInterface):
    input_spec = _PlotFDInputSpec
    output_spec = _PlotFDOutputSpec

    def _run_interface(self, runtime):
        fd_values = np.loadtxt(self.inputs.in_fd, skiprows=1)
        times = np.array(self.inputs.metadata['FrameTimesStart'])
        durations = np.array(self.inputs.metadata['FrameDuration'])
        midframe_times = times[:-1] + durations[:-1] / 2

        mask = midframe_times >= 120
        midframe_times = midframe_times[mask]
        fd_values = fd_values[mask]

        setup_plot_style()

        sns.lineplot(x=midframe_times, y=fd_values, marker='o', linewidth=2.5, color='crimson')

        plt.xlabel('Time (s)', fontsize=20, fontweight='bold')
        plt.ylabel('Framewise Displacement (mm)', fontsize=20, fontweight='bold')
        plt.title('FD plot for PET QC', fontsize=22, fontweight='bold', pad=20)
        plt.xticks(fontsize=16, fontweight='bold')
        plt.yticks(fontsize=16, fontweight='bold')

        sns.despine(trim=True)
        plt.tight_layout()

        output_filename = os.path.abspath('fd_plot.png')
        plt.savefig(output_filename, bbox_inches='tight')
        plt.close()

        self._results['out_file'] = output_filename
        return runtime


class _PlotRotationInputSpec(BaseInterfaceInputSpec):
    mot_param = File(exists=True, mandatory=True, desc="motion parameters")
    in_file = File(exists=True, mandatory=True, desc="File to be plotted")
    metadata = traits.Dict(mandatory=True, desc='Metadata dictionary containing timing info')
    out_file = traits.File(exists=False, desc="output file name")


class _PlotRotationOutputSpec(TraitedSpec):
    out_file = File(desc="Output file")


class PlotRotation(SimpleInterface):
    input_spec = _PlotRotationInputSpec
    output_spec = _PlotRotationOutputSpec

    def _run_interface(self, runtime):
        motion = np.loadtxt(self.inputs.mot_param)
        times = np.array(self.inputs.metadata['FrameTimesStart'])
        durations = np.array(self.inputs.metadata['FrameDuration'])
        midframe_times = times + durations / 2

        mask = midframe_times >= 120
        midframe_times = midframe_times[mask]
        rot_angles = motion[mask, :3]

        rotation_df = pd.DataFrame({
            'Time (s)': np.tile(midframe_times, 3),
            'Rotation (degrees)': rot_angles.flatten(),
            'Axis': np.repeat(['X', 'Y', 'Z'], len(midframe_times))
        })

        setup_plot_style()

        sns.lineplot(
            data=rotation_df,
            x='Time (s)',
            y='Rotation (degrees)',
            hue='Axis',
            marker='o',
            linewidth=2.5,
            palette='tab10'
        )

        plt.xlabel('Time (s)', fontsize=20, fontweight='bold')
        plt.ylabel('Rotation (degrees)', fontsize=20, fontweight='bold')
        plt.title('Rotation plot for PET QC', fontsize=22, fontweight='bold', pad=20)
        plt.xticks(fontsize=16, fontweight='bold')
        plt.yticks(fontsize=16, fontweight='bold')
        plt.legend(title='Axis', fontsize=14, title_fontsize=16, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3, frameon=False)

        sns.despine(trim=True)
        plt.tight_layout(rect=[0, 0.1, 1, 1])

        output_filename = os.path.abspath('rotation_plot.png')
        plt.savefig(output_filename, bbox_inches='tight')
        plt.close()

        self._results['out_file'] = output_filename
        return runtime


class _PlotTranslationInputSpec(BaseInterfaceInputSpec):
    mot_param = File(exists=True, mandatory=True, desc="motion parameters")
    in_file = File(exists=True, mandatory=True, desc="File to be plotted")
    metadata = traits.Dict(mandatory=True, desc='Metadata dictionary containing timing info')
    out_file = traits.File(exists=False, desc="output file name")


class _PlotTranslationOutputSpec(TraitedSpec):
    out_file = File(desc="Output file")


class PlotTranslation(SimpleInterface):
    input_spec = _PlotTranslationInputSpec
    output_spec = _PlotTranslationOutputSpec

    def _run_interface(self, runtime):
        motion = np.loadtxt(self.inputs.mot_param)
        times = np.array(self.inputs.metadata['FrameTimesStart'])
        durations = np.array(self.inputs.metadata['FrameDuration'])
        midframe_times = times + durations / 2

        mask = midframe_times >= 120
        midframe_times = midframe_times[mask]
        translations = motion[mask, 3:6]

        translation_df = pd.DataFrame({
            'Time (s)': np.tile(midframe_times, 3),
            'Translation (mm)': translations.flatten(),
            'Axis': np.repeat(['X', 'Y', 'Z'], len(midframe_times))
        })

        setup_plot_style()

        sns.lineplot(
            data=translation_df,
            x='Time (s)',
            y='Translation (mm)',
            hue='Axis',
            marker='o',
            linewidth=2.5,
            palette='tab10'
        )

        plt.xlabel('Time (s)', fontsize=20, fontweight='bold')
        plt.ylabel('Translation (mm)', fontsize=20, fontweight='bold')
        plt.title('Translation plot for PET QC', fontsize=22, fontweight='bold', pad=20)
        plt.xticks(fontsize=16, fontweight='bold')
        plt.yticks(fontsize=16, fontweight='bold')
        plt.legend(title='Axis', fontsize=14, title_fontsize=16, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3, frameon=False)

        sns.despine(trim=True)
        plt.tight_layout(rect=[0, 0.1, 1, 1])

        output_filename = os.path.abspath('translation_plot.png')
        plt.savefig(output_filename, bbox_inches='tight')
        plt.close()

        self._results['out_file'] = output_filename
        return runtime
    

def generate_tac_figures(tacs_tsv, metadata, output_dir=None): 
    import matplotlib.pyplot as plt
    import os
    import pandas as pd
    import seaborn as sns
    import re
    from pathlib import Path
    # Default to the current directory if output_dir is None
    if output_dir is None:
        output_dir = os.getcwd()

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Load the data
    tac_data = pd.read_csv(tacs_tsv, sep='\t')

    # Calculate midframe times
    tac_data['midframe'] = (tac_data['frame_times_start'] + tac_data['frame_times_end']) / 2

    region_cols = [col for col in tac_data.columns if col not in ['frame_times_start', 'frame_times_end', 'midframe']]

    def average_lr_regions(df, region_columns):
        averaged_data = pd.DataFrame()
        processed_regions = set()

        for col in region_columns:
            base_name = re.sub(r'(_L|_R)$', '', col)

            if base_name in processed_regions:
                continue

            left_col = f'{base_name}_L'
            right_col = f'{base_name}_R'

            if left_col in df and right_col in df:
                averaged_data[base_name] = df[[left_col, right_col]].mean(axis=1)
            else:
                averaged_data[base_name] = df[col]

            processed_regions.add(base_name)

        return averaged_data

    avg_tac_data = average_lr_regions(tac_data, region_cols)
    avg_tac_data['midframe'] = tac_data['midframe']

    tac_melted = avg_tac_data.melt(
        id_vars=['midframe'],
        var_name='Region',
        value_name='Uptake'
    )

    cortical_regions = [
        col for col in avg_tac_data.columns if any(keyword in col.lower() for keyword in [
            'gyrus', 'cortex', 'cingulate', 'frontal', 'temporal', 'parietal', 'occipital', 'insula', 'cuneus'
        ])
    ]

    subcortical_regions = [
        col for col in avg_tac_data.columns if any(keyword in col.lower() for keyword in [
            'caudate', 'putamen', 'thalamus', 'pallidum', 'accumbens', 'amygdala', 'hippocampus'
        ])
    ]

    ventricular_regions = [
        col for col in avg_tac_data.columns if 'ventricle' in col.lower()
    ]

    other_regions = [
        col for col in avg_tac_data.columns if col not in cortical_regions + subcortical_regions + ventricular_regions + ['midframe']
    ]

    unit = metadata.get('Units', 'Uptake')

    def plot_regions(df, regions, title, unit, output_path):
        plt.figure(figsize=(16, 10))

        sns.set(style='whitegrid', font_scale=1.4, context='talk')
        palette = sns.color_palette('tab10', n_colors=len(regions))

        plot = sns.lineplot(
            data=df[df['Region'].isin(regions)],
            x='midframe',
            y='Uptake',
            hue='Region',
            marker='o',
            linewidth=2.5,
            markersize=8,
            palette=palette
        )

        plot.set_xlabel('Time (s)', fontsize=20, fontweight='bold')
        plot.set_ylabel(f'Uptake ({unit})', fontsize=20, fontweight='bold')
        plot.set_title(title, fontsize=22, fontweight='bold', pad=20)

        plt.xticks(fontsize=16, fontweight='bold')
        plt.yticks(fontsize=16, fontweight='bold')

        plt.legend(
            title='Region',
            bbox_to_anchor=(0.5, -0.15),
            loc='upper center',
            fontsize=14,
            title_fontsize=16,
            frameon=False,
            ncol=3
        )

        sns.despine(trim=True)

        plt.tight_layout(rect=[0, 0.1, 1, 1])
        plt.savefig(output_path)
        plt.close()

    figures = []
    region_groups = [
        (cortical_regions, 'Cortical Regions TACs'),
        (subcortical_regions, 'Subcortical Regions TACs'),
        (ventricular_regions, 'Ventricular Regions TACs'),
        (other_regions, 'Other Regions TACs')
    ]

    for regions, title in region_groups:
        if regions:
            fig_filename = title.replace(' ', '_').lower() + '.png'
            fig_path = os.path.join(output_dir, fig_filename)
            plot_regions(tac_melted, regions, title, unit, fig_path)
            figures.append(fig_path)

    return figures