"""Regression coverage for authenticated download diagnostics."""

import subprocess
import traceback
from unittest.mock import patch

from mriqc.bin.abide2bids import fetch


def test_download_failure_redacts_credentials(tmp_path):
    """Retain curl's failure status without displaying the NITRC credential."""
    user = 'synthetic-nitrc-user'
    password = 'synthetic-nitrc-secret-42'

    def fail_download(command):
        raise subprocess.CalledProcessError(60, command)

    with patch('tempfile.mkdtemp', return_value=str(tmp_path)), patch(
        'subprocess.check_call', side_effect=fail_download
    ) as command:
        try:
            fetch(('https://example.invalid/NITRC_IR_synthetic', user, password, str(tmp_path)))
        except subprocess.CalledProcessError as error:
            assert error.returncode == 60
            diagnostic = str(error) + ''.join(traceback.format_exception(error))
            assert user not in diagnostic
            assert password not in diagnostic
            assert '<arguments redacted>' in diagnostic
            assert user not in repr(error.cmd)
            assert password not in repr(error.cmd)
        else:
            raise AssertionError('The download failure was not propagated')
    command.assert_called_once()
