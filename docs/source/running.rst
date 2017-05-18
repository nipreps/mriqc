
.. _running_mriqc:

Running mriqc
-------------

Command line interface
^^^^^^^^^^^^^^^^^^^^^^

.. argparse::
   :ref: mriqc.bin.mriqc_run.get_parser
   :prog: mriqc
   :nodefault:
   :nodefaultconst:


"Bare-metal" installation (Python 2/3)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The software automatically finds the data the input folder if it follows the
:abbr:`BIDS (brain imaging data structure)` standard [BIDS]_.
A fast and easy way to check that your dataset fulfills the
:abbr:`BIDS (brain imaging data structure)` standard is
the `BIDS validator <http://incf.github.io/bids-validator/>`_.

Since ``mriqc`` follows the [BIDSApps]_ specification, the execution is
split in two consecutive steps: a first level (or ``participant``) followed
by a second level (or ``group`` level).
In the ``participant`` level, all individual images to be processed are run
through the pipeline, and the :ref:`MRIQC measures <measures>` are extracted and
the individual reports (see :ref:`The MRIQC Reports <reports>`) generated.
In the ``group`` level, the :abbr:`IQMs (image quality metrics)` extracted in
first place are combined in a table and the group reports are generated.

The first (``participant``) level is executed as follows: ::

  mriqc bids-dataset/ out/ participant


Please note the keyword ``participant`` as fourth positional argument.
It is possible to run ``mriqc`` on specific subjects using ::

  mriqc bids-dataset/ out/ participant --participant_label S001 S002

where ``S001`` and ``S002`` are subject identifiers, corresponding to the folders
``sub-S001`` and ``sub-S002`` in the :abbr:`BIDS (brain imaging data structure)` tree.
Here, it is also accepted to use the ``sub-`` prefix ::

  mriqc bids-dataset/ out/ participant --participant_label sub-S001 sub-S002


.. note::

   If the argument :code:`--participant_label` is not provided, then all
   subjects will be processed and the group level analysis will
   automatically be executed without need of running the command in item 3.

After running the ``participant`` level with the :code:`--participant_label` argument,
the ``group`` level will not be automatically triggered.
To run the group level analysis: ::

  mriqc bids-dataset/ out/ group


Examples of the generated visual reports are found in `mriqc.org <http://mriqc.org>`_.


Depending on the input images, the resulting outputs will vary as described next.


Containerized versions
^^^^^^^^^^^^^^^^^^^^^^

If you have Docker installed, the quickest way to get ``mriqc`` to work
is following :ref:`the running with docker guide <docker>`.

Running MRIQC on HPC clusters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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

