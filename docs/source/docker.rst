
Run mriqc with docker
=====================


Preliminaries
-------------

#. Get the Docker Engine (https://docs.docker.com/engine/installation/)
#. Have your data organized in BIDS
   (`get BIDS specification <http://bids.neuroimaging.io/>`_,
   `see BIDS paper <http://dx.doi.org/10.1038/sdata.2016.44>`_).
#. Validate your data (http://incf.github.io/bids-validator/). You can
   safely use the BIDS-validator since no data is uploaded to the server,
   works locally in your browser.




Running mriqc
-------------


1. Test that the mriqc container works correctly. A successful run will show 
   the current mriqc version in the last line of the output:

  ::

      
      docker run -it poldracklab/mriqc:latest -v


2. Run the participants level in subjects 001 002 003:

  ::

      
      docker run -v <bids_dir>:/data -v <scratch_dir>:/scratch -w /scratch poldracklab/mriqc:latest /data /scratch/out participant --participant_level 001 002 003 -w /scratch/work


3. Run the group level and report generation on previously processed
   subjects:

  ::

      
      docker run -v <bids_dir>:/data -v <scratch_dir>:/scratch -w /scratch poldracklab/mriqc:latest /data /scratch/out group -w /scratch/work


.. note::

   If the argument :code:`--participant_level` is not provided, then all
   subjects will be processed and the group level analysis will
   automatically be executed without need of running the command in item 3.



Explaining the mriqc-docker command line
----------------------------------------

Let's dissect this command line:


+ :code:`docker run`- instructs the docker engine to get and run certain
  image (which is the last of docker-related arguments:
  :code:`poldracklab/mriqc:latest`)
+ :code:`-v <bids_dir>:/data` - instructs docker to mount the local
  directory `<bids_dir>`into :code:`/data` inside the container.
+ :code:`-v <scratch_dir>:/scratch`- instructs docker to mount the local
  directory `<work_dir>`into :code:`/scratch` inside the container.
+ :code:`poldracklab/mriqc:latest` - is the name of the image to be run, and
  sets an endpoint for the docker-related arguments in the command line.
+ :code:`/data /scratch/out participant -w /scratch/work` - are the standard
  arguments of mriqc.

