
Installing *MRIQC*
******************
*MRIQC* is a *NiPreps* (`www.nipreps.org <https://nipreps.org>`__)
*BIDS App* [BIDSApps]_.
As such, *MRIQC* can be installed manually (*Bare-metal installation*,
see below) or containerized.
For containerized execution with *Docker* or *Singularity*, please
follow the documentation on the *NiPreps* site
(`introduction <https://www.nipreps.org/apps/framework/>`__).

"Bare-metal" installation
-------------------------
If, for some reason, you really need a custom installation,
*MRIQC* can be installed as follows.
First, please make sure you have the execution system dependencies
installed (see below).
Second, the latest development version of MRIQC can be installed from
github using ``pip`` on a Python 3 environment::

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

  sudo apt-get install afni ants
  sudo ln -sf /usr/lib/ants/N4BiasFieldCorrection /usr/local/bin/

After installation, make sure that all the necessary binaries are added to the ``$PATH`` environment
variable, for the user that will run ``mriqc``.

Otherwise, you can follow each software installation guide:
`AFNI <https://afni.nimh.nih.gov/afni/doc/howto/0>`_,
and `ANTs <http://stnava.github.io/ANTs/>`_.

.. warning::

    Please note that *MRIQC* 22.0.0 and later requires Freesurfer's *SynthStrip* tool.

Please also install *FreeSurfer* (above 7.2) using `their guidelines <https://surfer.nmr.mgh.harvard.edu/fswiki/DownloadAndInstall>`__.
If your *FreeSurfer* version is older than 7.2, then you can get *MRIQC*'s requirements with::

  wget https://github.com/freesurfer/freesurfer/blob/dev/mri_synthstrip/synthstrip.1.pt -P ${FREESURFER_HOME}/models/

.. danger::

        You will get the following error if you do not install *SynthStrip*'s requirement::

          The 'model' trait of a _SynthStripInputSpec instance must be a pathlike object or string representing an existing file, but a value of '<undefined>' <class 'str'> was specified.`

        If the *SynthStrip* requirement was downloaded, please make sure your environment has defined the variable ``$FREESURFER_HOME`` and that it is pointing at the right directory::

          echo $FREESURFER_HOME

        If the ``$FREESURFER_HOME`` environment variable is defined, check whether the model file is available at the expected path::

          $ stat $FREESURFER_HOME/models/synthstrip.1.pt 
            File: /opt/freesurfer/models/synthstrip.1.pt
            Size: 30851709    Blocks: 60264      IO Block: 4096   regular file
          Device: fd00h/64768d  Inode: 12583379    Links: 1
          Access: (0644/-rw-r--r--)  Uid: (    0/    root)   Gid: (    0/    root)
          Access: 2024-04-09 11:13:29.091893354 +0200
          Modify: 2023-01-20 09:39:54.284056264 +0100
          Change: 2023-01-20 09:39:54.284056264 +0100
           Birth: 2023-01-20 09:39:54.224056213 +0100

        If *SynthStrip*'s model file is not present, the output will look like::

          $ stat $FREESURFER_HOME/models/synthstrip.1.pt
          stat: cannot statx '/opt/freesurfer/models/synthstrip.1.pt': No such file or directory
