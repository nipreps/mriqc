"""Manipulate Python warnings."""
import warnings
import logging

_wlog = logging.getLogger("py.warnings")
_wlog.addHandler(logging.NullHandler())


def _warn(message, category=None, stacklevel=1, source=None):
    """Redefine the warning function."""
    if category is not None:
        category = type(category).__name__
        category = category.replace("type", "WARNING")

    logging.getLogger("py.warnings").warning(f"{category or 'WARNING'}: {message}")


def _showwarning(message, category, filename, lineno, file=None, line=None):
    _warn(message, category=category)


warnings.warn = _warn
warnings.showwarning = _showwarning

# warnings.filterwarnings("ignore", category=FutureWarning)
# warnings.filterwarnings("ignore", category=DeprecationWarning)
# warnings.filterwarnings("ignore", category=ResourceWarning)
# # cmp is not used by mriqc, so ignore nipype-generated warnings
# warnings.filterwarnings("ignore", "cmp not installed")
# warnings.filterwarnings(
#     "ignore", "This has not been fully tested. Please report any failures."
# )
# warnings.filterwarnings("ignore", "sklearn.externals.joblib is deprecated in 0.21")
# warnings.filterwarnings("ignore", "can't resolve package from __spec__ or __package__")
