
Installation
============
Containerized versions
----------------------
If you have Docker installed, the quickest way to get ``mriqc`` to work
is following :ref:`the running with docker guide <docker>`.

We recommend trying containerized versions first to avoid installation
issues.
MRIQC uses bleeding-edge (oftentimes unreleased) versions of
``nipype`` and ``niworkflows`` and "bare-metal" installations can
be hard.
Nonetheless, we offer support on our `github repository
<https://github.com/nipreps/mriqc/issues>`_.


"Bare-metal" installation
-------------------------
If, for some reason, you really need a bare-metal installation,
MRIQC can be installed as follows.
First, please make sure you have the execution system dependencies
installed (see below).
Second, the latest development version of MRIQC can be installed from
github using ``pip`` on a Python 3 environment: ::

  python -m pip install -U mriqc


.. warning::

        As of MRIQC 0.9.4, Python 2 is no longer supported.

.. warning::

        MRIQC uses matplotlib to create graphics. By default, matplotlib is configured to
        plot through an interactive Tcl/tk interface, which requires a functional display to be available.
        In *head-less* settings (for example, when running under tmux),
        you may see an error::

                _tkinter.TclError: couldn't connect to display "localhost:10.0"

        There are two pathways to fix this issue.
        One is setting up a virtual display with a tool like XVfb.
        Alternatively, you can configure your matplotlib distribution to perform on
        head-less mode by default.
        That is achieved by uncommenting the ``backend : Agg`` line in the matplotlib's
        configuration file.
        The location of the configuration file can be retrieved with Python::

          >>> import matplotlib
          >>> print(matplotlib.matplotlib_fname())

        Alternatively, you can issue the following shell command-line to edit this setting::

        $ sed -i 's/\(backend *: \).*$/\1Agg/g' $( python -c "import matplotlib; print(matplotlib.matplotlib_fname())" )



Execution system dependencies
.............................
If you are using a a `Neurodebian <http://neuro.debian.net/>`_ Linux distribution,
installation should be as easy as::

  sudo apt-get install fsl afni ants
  sudo ln -sf /usr/lib/ants/N4BiasFieldCorrection /usr/local/bin/

After installation, make sure that all the necessary binaries are added to the ``$PATH`` environment
variable, for the user that will run ``mriqc``.

Otherwise, you can follow each software installation guide:
`FSL <http://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FslInstallation>`_,
`AFNI <https://afni.nimh.nih.gov/afni/doc/howto/0>`_,
and `ANTs <http://stnava.github.io/ANTs/>`_.
