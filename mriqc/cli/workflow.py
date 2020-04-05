"""
The workflow builder factory method.

All the checks and the construction of the workflow are done
inside this function that has pickleable inputs and output
dictionary (``retval``) to allow isolation using a
``multiprocessing.Process`` that allows dmriprep to enforce
a hard-limited memory-scope.
"""


def build_workflow(config_file, retval):
    """Create the Nipype Workflow that supports the whole execution graph."""
    from .. import config
    from ..workflows.core import init_mriqc_wf

    config.load(config_file)
    retval["return_code"] = 1
    retval["workflow"] = None

    retval["workflow"] = init_mriqc_wf()
    retval["return_code"] = int(retval["workflow"] is None)
    return retval
