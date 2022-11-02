import migas

from .. import __version__, config


def setup_migas(init: bool = True) -> None:
    """
    Prepare the migas python client to communicate with a migas server.
    If ``init`` is ``True``, send an initial breadcrumb.
    """
    # generate session UUID from generated run UUID
    session_id = None
    if config.execution.run_uuid:
        session_id = config.execution.run_uuid.split('_', 1)[-1]

    migas.setup(session_id=session_id)
    if init:
        # send initial status ping
        send_breadcrumb(status='R', status_desc='workflow start')


def send_breadcrumb(**kwargs) -> dict:
    """
    Communicate with the migas telemetry server. This requires `migas.setup()` to be called.
    """
    res = migas.add_project("nipreps/mriqc", __version__, **kwargs)
    return res
