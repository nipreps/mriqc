import importlib
from pathlib import Path

import pytest

# Skip tests if required packages are missing
np = pytest.importorskip('numpy')
nb = pytest.importorskip('nibabel')

from mriqc.interfaces.pet import ChooseRefHMC, FDStats


@pytest.mark.parametrize('n_frames', [5, 3])
def test_choose_ref_hmc(tmp_path, n_frames):
    """Check that the frame with maximum intensity is selected."""
    data = np.random.rand(2, 2, 2, n_frames)
    frame_sums = data.sum(axis=(0, 1, 2))
    max_idx = int(np.argmax(frame_sums))

    img = nb.Nifti1Image(data, np.eye(4))
    in_file = tmp_path / 'in_pet.nii.gz'
    img.to_filename(in_file)

    result = ChooseRefHMC(in_file=str(in_file)).run()
    out_file = Path(result.outputs.out_file)
    assert out_file.exists()

    # Ensure the interface reports the correct reference frame index
    assert result.outputs.ref_frame == max_idx

    out_img = nb.load(out_file)
    assert np.allclose(out_img.get_fdata(), data[..., max_idx])


def test_fdstats(tmp_path):
    """FDStats should compute mean, count and percentage correctly."""
    fd_values = [0.1, 0.3, 0.15]
    in_fd = tmp_path / 'fd.txt'
    in_fd.write_text('FD\n' + '\n'.join(str(v) for v in fd_values))

    res = FDStats(in_fd=str(in_fd)).run()
    out = res.outputs.out_fd

    expected_mean = float(np.mean(fd_values))
    expected_num = int(np.sum(np.array(fd_values) > 0.2))
    expected_perc = float(expected_num * 100 / (len(fd_values) + 1))

    assert out['mean'] == pytest.approx(expected_mean)
    assert out['num'] == expected_num
    assert out['perc'] == pytest.approx(expected_perc)


@pytest.mark.skipif(
    any(importlib.util.find_spec(pkg) is None for pkg in ('pandas', 'matplotlib', 'seaborn')),
    reason='Required plotting libraries are not installed.'
)
def test_generate_tac_figures(tmp_path):
    """generate_tac_figures should create expected figure files."""
    import pandas as pd
    from mriqc.qc.pet import generate_tac_figures

    # Create example TAC data
    df = pd.DataFrame({
        'frame_times_start': [0, 10],
        'frame_times_end': [10, 20],
        'region_L': [1.0, 2.0],
        'region_R': [1.5, 2.5],
    })
    tsv = tmp_path / 'tac.tsv'
    df.to_csv(tsv, sep='\t', index=False)

    meta = {'Units': 'kBq'}
    figs = generate_tac_figures(str(tsv), meta, output_dir=str(tmp_path))

    assert figs, 'No figures were generated.'
    for fig in figs:
        assert Path(fig).exists()


@pytest.mark.skipif(
    any(importlib.util.find_spec(pkg) is None for pkg in ('bids', 'niworkflows')),
    reason='Required packages are not installed.'
)
def test_tracer_entity_derivatives(tmp_path):
    """DerivativesDataSink should keep tracer label in output filenames."""
    from mriqc import config
    from mriqc.testing import mock_config
    from mriqc.utils.misc import initialize_meta_and_data
    from mriqc.interfaces import DerivativesDataSink

    pet_file = (
        Path(__file__).parent.parent
        / 'data'
        / 'pet'
        / 'sub-01'
        / 'ses-baseline'
        / 'pet'
        / 'sub-01_ses-baseline_trc-ps13_pet.nii.gz'
    )

    with mock_config():
        config.execution.bids_dir = pet_file.parents[3]
        config.execution.init()
        config.workflow.inputs = {'pet': [str(pet_file)]}
        initialize_meta_and_data()
        assert config.workflow.inputs_entities['pet'][0]['trc'] == 'ps13'

        ds = DerivativesDataSink(
            base_directory=tmp_path,
            datatype='figures',
            desc='carpet',
            extension='.svg',
            dismiss_entities=('part',),
        )
        ds.inputs.in_file = str(pet_file)
        ds.inputs.source_file = str(pet_file)
        result = ds.run()

    assert 'trc-ps13' in Path(result.outputs.out_file).name