"""Tests for the BIDS utilities."""

from mriqc.utils.bids import write_bidsignore


def test_write_bidsignore_excludes_derived_outputs(tmp_path):
    write_bidsignore(tmp_path)
    patterns = (tmp_path / '.bidsignore').read_text().splitlines()

    # MRIQC writes reportlets under figures/ and per-run timeseries
    assert 'figures/' in patterns
    assert '*_timeseries.tsv' in patterns
    assert '*_timeseries.json' in patterns
