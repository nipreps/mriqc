
Installation
------------

We have included ``mriqc`` in the PyPi resource ::

  pip install mriqc


Execution system dependencies
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you are using a linux distribution with `neurodebian <http://neuro.debian.net/>`_, installation
should be as easy as::

  sudo apt-get install fsl afni ants
  sudo ln -sf /usr/lib/ants/N4BiasFieldCorrection /usr/local/bin/

After installation, make sure that all the necessary binaries are added to the ``$PATH`` environment
variable, for the profile used run ``mriqc``.

Otherwise, you can follow each software installation guide: `FSL <http://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FslInstallation>`_, `AFNI <https://afni.nimh.nih.gov/afni/doc/howto/0>`_, and `ANTs <http://stnava.github.io/ANTs/>`_.
