
Installation
------------

Containerized versions
^^^^^^^^^^^^^^^^^^^^^^

If you have Docker installed, the quickest way to get ``mriqc`` to work
is following :ref:`the running with docker guide <docker>`.

We recommend trying containerized versions first to avoid installation
issues.
MRIQC uses bleeding-edge (oftentimes unreleased) versions of 
``nipype`` and ``niworkflows`` and "bare-metal" installations can
be hard.
Nonetheless, we offer support on our `github repository
<https://github.com/poldracklab/mriqc/issues>`_.


"Bare-metal" installation (only Python 3)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If, for some reason, you really need a bare-metal installation,
MRIQC can be installed as follows.
First, please make sure you have the execution system dependencies
installed (see below).
Second, the latest development version of MRIQC can be installed from
github using ``pip`` on a Python 3 environment: ::

  pip install -r https://raw.githubusercontent.com/poldracklab/mriqc/master/requirements.txt
  pip install git+https://github.com/poldracklab/mriqc.git


.. warning::

	As of MRIQC 0.9.4, Python 2 is no longer supported.


Execution system dependencies
'''''''''''''''''''''''''''''

If you are using a linux distribution with `neurodebian <http://neuro.debian.net/>`_, installation
should be as easy as::

  sudo apt-get install fsl afni ants
  sudo ln -sf /usr/lib/ants/N4BiasFieldCorrection /usr/local/bin/

After installation, make sure that all the necessary binaries are added to the ``$PATH`` environment
variable, for the user that will run ``mriqc``.

Otherwise, you can follow each software installation guide: 
`FSL <http://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FslInstallation>`_, 
`AFNI <https://afni.nimh.nih.gov/afni/doc/howto/0>`_, 
and `ANTs <http://stnava.github.io/ANTs/>`_.
