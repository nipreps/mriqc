

Running mriqc
-------------

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
through the pipeline, and the :ref:`MRIQC measures` are extracted and
the individual reports (see :ref:`The MRIQC Reports`) generated.
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
is following :ref:`Run mriqc with docker`.

Running MRIQC on HPC clusters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Singularity containers
......................

Requesting resources
....................



.. topic:: References

  .. [BIDS] `Brain Imaging Data Structure <http://bids.neuroimaging.io/>`_
  .. [BIDSApps] `BIDS-Apps: portable neuroimaging pipelines that understand BIDS
     datasets <http://bids-apps.neuroimaging.io/>`_

