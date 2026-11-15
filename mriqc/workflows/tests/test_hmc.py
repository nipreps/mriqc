# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# Copyright 2026 The NiPreps Developers <nipreps@gmail.com>
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
"""Head-motion correction: resampling the BOLD series, and the engines."""

import nibabel as nb
import numpy as np
import pytest
from nitransforms.linear import load
from nitransforms.resampling import apply

from mriqc.workflows.functional.base import _apply_transforms


def test_apply_transforms_in_memory(tmp_path, monkeypatch):
    """Resampling the series loaded in memory gives the output of reading it from disk."""
    rng = np.random.default_rng(1234)
    nvols = 6
    in_file = tmp_path / 'sub-01_task-rest_bold.nii.gz'
    data = rng.normal(100, 10, (10, 12, 8, nvols)).astype('float32')
    nb.Nifti1Image(data, np.diag([3.0, 3.0, 3.3, 1.0])).to_filename(in_file)

    # 3dvolreg's -1Dmatrix_save format: one row per volume with a 3x4 matrix
    xfms = np.tile(np.eye(4)[:3].ravel(), (nvols, 1))
    xfms[:, 3] = np.linspace(0, 0.6, nvols)
    xfms[:, 7] = np.linspace(0, -0.4, nvols)
    xfm_file = tmp_path / 'hmc.aff12.1D'
    np.savetxt(xfm_file, xfms)

    expected = apply(
        load(xfm_file, fmt='afni', reference=in_file, moving=in_file),
        in_file,
        dtype_width=4,
        serialize_nvols=2,
        max_concurrent=2,
        mode='reflect',
    )

    monkeypatch.chdir(tmp_path)
    out_file = _apply_transforms(str(in_file), str(xfm_file), 2)
    realigned = np.asanyarray(nb.load(out_file).dataobj)
    assert realigned.shape == data.shape
    assert not np.array_equal(realigned, data)  # the transforms moved the volumes
    assert np.array_equal(realigned, np.asanyarray(expected.dataobj))


@pytest.mark.parametrize('engine', ['3dvolreg', 'niimath'])
def test_hmc_engine(monkeypatch, engine):
    """--hmc-engine niimath corrects BOLD with niimath, and leaves DWI on 3dvolreg."""
    from nipype.interfaces.afni import Volreg

    from mriqc import config
    from mriqc.interfaces.niimath import Moco
    from mriqc.testing import mock_config
    from mriqc.workflows.diffusion.base import hmc_workflow
    from mriqc.workflows.functional.base import hmc
    from mriqc.workflows.utils import get_hmc

    with mock_config():
        monkeypatch.setattr(config.workflow, 'hmc_engine', engine)
        # mock_config() gives one size for all modalities
        monkeypatch.setattr(config.workflow, 'biggest_file_gb', {'bold': 0.1, 'dwi': 0.1})
        monkeypatch.setattr(config.nipype, 'omp_nthreads', 3)
        bold = hmc().get_node('estimate_hm')
        dwi = hmc_workflow().get_node('motion_correct')
        assert isinstance(get_hmc(modality='dwi'), Volreg)

    assert isinstance(bold.interface, Moco if engine == 'niimath' else Volreg)
    assert bold.n_procs == (3 if engine == 'niimath' else 1)
    assert isinstance(dwi.interface, Volreg)
    assert dwi.n_procs == 1


def test_hmc_engine_cli(tmp_path):
    """--hmc-engine is listed in --help, and parsed."""
    from mriqc.cli.parser import _build_parser

    parser = _build_parser()
    assert '--hmc-engine' in parser.format_help()
    args = [str(tmp_path), str(tmp_path / 'out'), 'participant', '--hmc-engine', 'niimath']
    assert parser.parse_args(args).hmc_engine == 'niimath'
    args[-1] = 'nifreeze'
    assert parser.parse_args(args).hmc_engine == 'nifreeze'
    assert '--dwi-eddy' in parser.format_help()
    assert not parser.parse_args(args).dwi_eddy
    assert parser.parse_args([*args, '--dwi-eddy']).dwi_eddy


@pytest.mark.parametrize('engine', ['3dvolreg', 'niimath'])
def test_dwi_eddy_requires_nifreeze(tmp_path, engine):
    """--dwi-eddy is an error with the engines that cannot correct eddy currents."""
    from mriqc.cli.parser import parse_args

    with pytest.raises(SystemExit):
        parse_args(
            [
                str(tmp_path),
                str(tmp_path / 'out'),
                'participant',
                '--hmc-engine',
                engine,
                '--dwi-eddy',
            ]
        )


@pytest.mark.parametrize('eddy', [False, True])
@pytest.mark.parametrize('seed', [None, 1234])
def test_nifreeze_engine(monkeypatch, seed, eddy):
    """--hmc-engine nifreeze corrects DWI with NiFreeze, and leaves BOLD on 3dvolreg."""
    from nipype.interfaces.afni import Volreg

    from mriqc import config
    from mriqc.interfaces.nifreeze import NiFreezeHMC
    from mriqc.testing import mock_config
    from mriqc.workflows.diffusion.base import hmc_workflow
    from mriqc.workflows.utils import get_hmc

    with mock_config():
        monkeypatch.setattr(config.workflow, 'hmc_engine', 'nifreeze')
        monkeypatch.setattr(config.workflow, 'biggest_file_gb', {'dwi': 0.41})
        monkeypatch.setattr(config.seeds, 'ants', seed)
        monkeypatch.setattr(config.workflow, 'dwi_eddy', eddy)
        monkeypatch.setattr(config.nipype, 'omp_nthreads', 3)
        assert isinstance(get_hmc(estimate=True), Volreg)
        wf = hmc_workflow()

    hmc = wf.get_node('motion_correct')
    assert isinstance(hmc.interface, NiFreezeHMC)
    assert hmc.n_procs == 3
    assert hmc.mem_gb == pytest.approx(22 * 0.41)
    assert (hmc.inputs.seed if seed else None) == seed
    assert hmc.inputs.eddy is eddy
    connected = {
        conn[1]
        for _, node, data in wf._graph.edges(data=True)
        if node is hmc
        for conn in data['connect']
    }
    assert connected == {'in_file', 'basefile', 'in_bvec', 'in_bval', 'brain_mask'}
