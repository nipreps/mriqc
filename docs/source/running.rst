
.. _running_mriqc:

Running mriqc
=============
.. tip::
     Try MRIQC online on `OpenNeuro <https://www.openneuro.org/>`_ - without
     installation!

MRIQC is a `BIDS-App <http://bids-apps.neuroimaging.io/>`_ [BIDSApps]_,
and therefore it inherently understands the :abbr:`BIDS (brain
imaging data structure)` standard [BIDS]_ and follows the
BIDS-Apps standard command line interface::

  mriqc bids-root/ output-folder/ participant

That simple command runs MRIQC on all the *T1w* and *BOLD* images found
under the BIDS-compliant folder ``bids-root/``.
The last ``participant`` keyword indicates that the first level analysis
is run. (i.e. extracting the :abbr:`IQMs (image quality metrics)` from the
images retrieved within ``bids-root/``).
The second level (``group``) is automatically run if no particular subject
is provided for analysis.

.. note::

   If the argument :code:`--participant-label` is not provided, then all
   subjects will be processed and the group level analysis will
   automatically be executed without need of running the command in item 3.

To specify one particular subject, the ``--participant-label`` argument
can be used::

  mriqc bids-root/ output-folder/ participant --participant-label S01 S02 S03

That command will run MRIQC only on the subjects indicated: only
``bids-root/sub-S01``, ``bids-root/sub-S02``, and ``bids-root/sub-S03``
will be processed.
In this case, the ``group`` level will not be triggered automatically.
We generate the ``group`` level results (the group level report and the
features CSV table) with: ::

  mriqc bids-root/ output-folder/ group

Examples of the generated visual reports are found
in :ref:`The MRIQC Reports <reports>`.

.. warning::

    MRIQC by default attempts to upload anonymized quality metrics to a publicly accessible
    web server (`mriqc.nimh.nih.gov <http://mriqc.nimh.nih.gov/>`_). The uploaded data consists
    only of calculated quality metrics and scanning parameters. It removes all personal
    health information and participant identifiers. We try to collect this data to build normal
    distributions for improved outlier detection, but if you do not wish to participate you can
    disable the submission with the ``--no-sub`` flag.

.. topic:: BIDS data organization

    The software automatically finds the data the input folder if it
    follows the :abbr:`BIDS (brain imaging data structure)` standard [BIDS]_.
    A fast and easy way to check that your dataset fulfills the
    :abbr:`BIDS (brain imaging data structure)` standard is
    the `BIDS validator <https://github.com/bids-standard/bids-validator>`_.

.. topic:: BIDS-App levels

    In the ``participant`` level, all individual images to be processed are run
    through the pipeline, and the :ref:`MRIQC measures <measures>` are extracted and
    the individual reports (see :ref:`The MRIQC Reports <reports>`) generated.
    In the ``group`` level, the :abbr:`IQMs (image quality metrics)` extracted in
    first place are combined in a table and the group reports are generated.

Command line interface
----------------------
.. argparse::
   :ref: mriqc.cli.parser._build_parser
   :prog: mriqc
   :nodefault:
   :nodefaultconst:

Running mriqc on HPC clusters
-----------------------------
Singularity containers
......................
Requesting resources
....................
We have profiled cores and memory usages with the *resource profiler*
tool of nipype.

An MRIQC run of one subject (from the ABIDE) dataset, containing only one
run, one BOLD task (resting-state) yielded the following report:

  .. raw:: html

      <iframe src="_static/bold-1subject-1task.html" height="345px" width="100%"></iframe>

  Using the ``MultiProc`` plugin of nipype with ``nprocs=10``, the workflow
  nodes run across the available processors for 41.68 minutes.
  A memory peak of 8GB is reached by the end of the runtime, when the
  plotting nodes are fired up.

We also profiled MRIQC on a dataset with 8 tasks (one run per task),
on ds030 of OpenfMRI:

  .. raw:: html

      <iframe src="_static/bold-1subject-8tasks.html" height="345px" width="100%"></iframe>

  Again, we used ``n_procs=10``. The software run for roughly about the same
  time (47.11 min). Most of the run time, memory usage keeps around a
  maximum of 10GB. Since we saw a memory consumption of 1-2GB during the
  the 1-task example, a rule of thumb may be that each task takes around
  1GB of memory.

.. topic:: References

  .. [BIDS] `Brain Imaging Data Structure <http://bids.neuroimaging.io/>`_
  .. [BIDSApps] `BIDS-Apps: portable neuroimaging pipelines that understand BIDS
     datasets <http://bids-apps.neuroimaging.io/>`_

.. include:: license.rst
