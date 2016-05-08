User Guide
==========

Contents
========

.. toctree::
   :maxdepth: 3

   installation
   running
   mriqc.reports


Installation
------------

We have included ``mriqc`` in the PyPi resource ::

  pip install mriqc


Dependencies
------------

If you are using a linux distribution with `neurodebian <http://neuro.debian.net/>`_, installation
should be as easy as::

  sudo apt-get fsl afni ants
  sudo ln -sf /usr/lib/ants/N4BiasFieldCorrection /usr/local/bin/

After installation, make sure that all the necessary binaries are added to the ``$PATH`` environment
variable, for the profile used run ``mriqc``.

Otherwise, you can follow each software installation guide: `FSL <http://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FslInstallation>`_, `AFNI <https://afni.nimh.nih.gov/afni/doc/howto/0>`_, and `ANTs <http://stnava.github.io/ANTs/>`_.

Running mriqc
-------------

The software automatically finds the data the input folder if it follows the
:abbr:`BIDS (brain imaging data structure)` standard [BIDS]_.
A fast and easy way to check that your dataset fulfills the
:abbr:`BIDS (brain imaging data structure)` standard is
the `BIDS validator <http://incf.github.io/bids-validator/>`_.

Running ``mriqc`` only requires: ::

  mriqc -B ~/Data/bids_dataset -o out/ -w work/


It is possible to run ``mriqc`` on specific subjects using ::

  mriqc -B ~/Data/bids_dataset -S S001 S002 -o out/ -w work/

where ``S001`` and ``S002`` are subject identifiers, corresponding to the folders
``sub-S001`` and ``sub-S002`` in the :abbr:`BIDS (brain imaging data structure)` tree. Here, it is also accepted to use the ``sub-`` prefix ::

  mriqc -B ~/Data/bids_dataset -S sub-S001 sub-S002 -o out/ -w work/


Depending on the input images, the resulting outputs will vary as described next.


  .. [BIDS] `Brain Imaging Data Structure <http://bids.neuroimaging.io/>`_
