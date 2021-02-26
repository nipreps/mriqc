# Copied shamelessly from
# https://github.com/dandi/dandi-cli/blob/da3b7a726c4a352dfb53a0c6bee59e660de827e6/dandi/utils.py#L49-L82
#
# This file is licensed under Apache 2, as noted in
# https://github.com/dandi/dandi-cli/blob/da3b7a726c4a352dfb53a0c6bee59e660de827e6/LICENSE
import sys


def is_interactive():
    """Return True if all in/outs are tty"""
    # TODO: check on windows if hasattr check would work correctly and add value:
    #
    return sys.stdin.isatty() and sys.stdout.isatty() and sys.stderr.isatty()


def setup_exceptionhook(ipython=False):
    """Overloads default sys.excepthook with our exceptionhook handler.
    If interactive, our exceptionhook handler will invoke
    pdb.post_mortem; if not interactive, then invokes default handler.
    """

    def _pdb_excepthook(type, value, tb):
        import traceback

        traceback.print_exception(type, value, tb)
        print()
        if is_interactive():
            import pdb

            pdb.post_mortem(tb)

    if ipython:
        from IPython.core import ultratb

        sys.excepthook = ultratb.FormattedTB(
            mode="Verbose",
            # color_scheme='Linux',
            call_pdb=is_interactive(),
        )
    else:
        sys.excepthook = _pdb_excepthook
