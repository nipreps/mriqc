
.. _containers:

Run mriqc with docker
*********************
Preliminaries
-------------
#. Get the Docker Engine (https://docs.docker.com/engine/installation/)
#. Have your data organized in BIDS
   (`get BIDS specification <https://bids.neuroimaging.io/>`_,
   `see BIDS paper <https://doi.org/10.1038/sdata.2016.44>`_).
#. Validate your data (https://bids-standard.github.io/bids-validator/).
   You can safely use the BIDS-validator since no data is uploaded to the
   server, it works locally in your browser.

.. warning ::

    `2GB is the default memory setting
    <https://docs.docker.com/docker-for-mac/>`_
    on a fresh installation of Docker for Mac.
    We recommend increasing the available memory for Docker containers

   When using docker with big datasets (+10GB) docker might fail.
   Changing the maximum size of the container will solve it: ::

    $ service docker stop
    $ dockerd --storage-opt dm.basesize=30G


.. warning ::

    On Windows installations, before using the ``-v`` switch to mount volumes into
    the container, it is necessary to `enable shared drives
    <https://docs.docker.com/docker-for-windows/#shared-drives>`_.



.. _docker_run_mriqc:

Running mriqc
-------------
1. Test that the mriqc container works correctly. A successful run will show
   the current mriqc version in the last line of the output:

  ::


      docker run -it nipreps/mriqc:latest --version


2. Run the :code:`participant` level in subjects 001 002 003:

  ::


      docker run -it --rm -v <bids_dir>:/data:ro -v <output_dir>:/out nipreps/mriqc:latest /data /out participant --participant_label 001 002 003


3. Run the group level and report generation on previously processed (use the same ``<output_dir>``)
   subjects:

  ::


      docker run -it --rm -v <bids_dir>:/data:ro -v <output_dir>:/out nipreps/mriqc:latest /data /out group


Explaining the mriqc docker command line
----------------------------------------
Let's dissect this command line:

+ :code:`docker run`- instructs the docker engine to get and run a certain
  image (which is the last of docker-related arguments:
  :code:`nipreps/mriqc:latest`)
+ :code:`-v <bids_dir>:/data:ro` - instructs docker to mount the local
  directory with your input dataset `<bids_dir>`into :code:`/data` inside
  the container in a read only mode.

.. warning ::

  If you are using Datalad to download your data from OpenNeuro or other sources, 
  you will need to remove the :code:`:ro` tag to allow the data to be downloaded into the 
  container. 


+ :code:`-v <output_dir>:/out`- instructs docker to mount the local
  directory `<output_dir>`into :code:`/out` inside the container. This is
  where the results of the QC analysis (reports, tables) will be stored.
+ :code:`nipreps/mriqc:latest` - this tells docker to run MRIQC. ``latest``
  corresponds to the version of MRIQC. You
  should replace ``latest`` with a version of MRIQC you want to use. Remember
  not to switch versions while analysing one dataset!
+ :code:`/data /scratch/out participant` - are the standard
  arguments of mriqc.

.. note::

   If the argument :code:`--participant_label` is not provided, then all
   subjects will be processed and the group level analysis will
   automatically be executed without need of running the command in item 3.

.. note::

    If you are using Datalad to download and manage your data, you will not be 
    able to use the ``:ro`` read-only tag unless you ``datalad get .`` your data prior to 
    running MRIQC. 

.. warning::

    Paths `<bids_dir>` and `<output_dir>` must be absolute.  In particular, specifying relative paths for
    `<output_dir>` will generate no error and mriqc will run to completion without error but produce no output.

.. warning::

    Uf possible, for security reasons, we recommend to run the docker command with the options
    ``--read-only --tmpfs /run --tmpfs /tmp``. This will run the docker image in
    read-only mode, and map the temporary folders ``/run`` and ``/tmp`` to the temporary
    folder of the host. This is not compatible with Datalad datasets.

Run mriqc with Apptainer/Singularity
************************************

For larger datasets, running MRIQC on a HPC may be necessary. Since Docker requires users to run as 
root, which isn't possible on HPCs, we use Apptainer here instead. Apptainer functions similarly to 
Docker, and is able to build Docker
containers from the Docker registry. 

Preliminaries
-------------
#. Ensure Apptainer is installed on your system: :code:`apptainer exec docker://alpine cat /etc/alpine-release`

#. Build your Apptainer container. 

#. Have your data organized in BIDS
   (`get BIDS specification <https://bids.neuroimaging.io/>`_,
   `see BIDS paper <https://doi.org/10.1038/sdata.2016.44>`_).

#. Validate your data (https://bids-standard.github.io/bids-validator/).
   You can safely use the BIDS-validator since no data is uploaded to the
   server, it works locally in your browser.


.. _singularity_write_sbatch:

Write job scheduler script
--------------------------
Most HPCs use some form of job scheduling software, such as Slurm or PGE, and require 
the use of job scripts to request resources and necessary variables. We provide an example
using SBATCH, originally written for Stanford's Sherlock. All lines appended with :code:`#TODO`
are lines that will need to be modified for your data and your HPC. Consult your 
HPC's documentation for further details.

Breaking down the script, all lines preceeded by :code:`#SBATCH` are commands to the scheduler 
to request resources, like time and memory. The :code:`--array` parameter runs multiple jobs per 
job script, and is generally recommended for running many jobs that do the same thing. To run all of
your participants, you need to set :code:`--array=1-n` where n is the number of particpants you have. 
To limit the number of jobs that will run concurrently, you can set :code:`--array=1-n%j` where j is the 
number of concurrent jobs. 

When the `--array` parameter is set, the Slurm job scheduler will run that number of array jobs.
It also sets a system variable $SLURM_ARRAY_TASK_ID that we will use with our BIDS participants.tsv
file to set a participant ID for the MRIQC command line interface. 

After we define all of our #SBATCH variables, we can move onto setting path variables ($STUDY, 
$BIDS_DIR, and $OUTPUT_DIR to pass to MRIQC. These should be absolute paths to your data. 

We use other variables to define the Apptainer command for our specific version of MRIQC, which 
should match the MRIQC container you built in the preliminary steps. 



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




