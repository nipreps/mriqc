"""Utilities and mocks for testing and documentation building."""
from contextlib import contextmanager
from pathlib import Path
from pkg_resources import resource_filename as pkgrf
from toml import loads
from tempfile import mkdtemp


@contextmanager
def mock_config():
    """Create a mock config for documentation and testing purposes."""
    from . import config

    filename = Path(pkgrf("mriqc", "data/config-example.toml"))
    settings = loads(filename.read_text())
    for sectionname, configs in settings.items():
        if sectionname != "environment":
            section = getattr(config, sectionname)
            section.load(configs, init=False)
    config.nipype.init()
    config.loggers.init()

    config.execution.work_dir = Path(mkdtemp())
    config.execution.bids_dir = Path(pkgrf("mriqc", "data/tests/ds000005")).absolute()
    config.execution.init()

    yield
