
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

.. _running-tests:

Running the tests
-----------------

Unit tests and doctests
.......................
*MRIQC*'s unit tests and doctests run with *pytest*.
From a clone of the repository, install *MRIQC* with the ``test`` extra::

  git clone https://github.com/nipreps/mriqc.git
  cd mriqc
  python -m pip install -e ".[test]"

Then, run the tests as the continuous integration does::

  pytest mriqc --doctest-modules --ignore=mriqc/bin --ignore=mriqc/interfaces/transitional.py

The *pytest* configuration in ``pyproject.toml`` stops at the first failure and turns
warnings into errors.
The continuous integration runs these tests on GitHub Actions
(``.github/workflows/pythonpackage.yml``) and, within the *Docker* image, on CircleCI
(``.circleci/config.yml``).

Integration tests
.................
To check that *MRIQC* behaves correctly end to end, the continuous integration runs
the *Docker* image on small datasets and compares the resulting IQMs with the
reference tables in ``mriqc/data/testdata/group_<modality>.tsv``:

* GitHub Actions (``.github/workflows/integration.yml``) processes DWI and FLAIR images,
  on linux/amd64 and linux/arm64.
  The datasets are assembled from pinned public sources by ``.github/testdata/fetch.py``
  (see ``.github/testdata/README.md``).
* CircleCI (``.circleci/config.yml``) processes the T1w and BOLD images of a small
  dataset hosted on OSF.

Each run first checks that the expected files were written
(``.github/integration/outputs-<modality>.txt`` and ``.circleci/circle_<modality>.txt``),
and then compares the group table with its reference using ``dfcheck``.
``dfcheck`` lists every IQM that differs from the reference (with a relative tolerance
of 1e-5), and fails if the columns or the number of images differ, or if an IQM
correlates below 0.95 with the reference across images.
On linux/amd64, the runs reproduce the reference exactly: *ANTs* runs single-threaded
(``--ants-nthreads 1``), and every run but BOLD fixes the random seed
(``--random-seed 42``; the BOLD IQMs do not depend on it).
Therefore, the continuous integration runs ``dfcheck --strict``, which also fails on any
value outside the tolerance, and a change that alters the IQMs must update the
reference table in the same pull request.
IQMs computed on linux/arm64 differ: GitHub Actions does not compare them, and CircleCI
compares them without ``--strict``.

For example, the DWI test can be reproduced locally from the root of the repository
with an image built from it::

  docker build -t nipreps/mriqc:ci .
  uv run --no-project --with nibabel --with numpy \
      python .github/testdata/fetch.py .github/testdata/dwi.json $PWD/testdata/dwi
  mkdir -p out work
  docker run --rm -u $(id -u):$(id -g) -e MRIQC_DEV=1 \
      -v $PWD/testdata/dwi:/data:ro -v $PWD/out:/out -v $PWD/work:/work \
      nipreps/mriqc:ci /data /out participant -m dwi --no-sub \
      --random-seed 42 --nprocs 4 --omp-nthreads 2 --ants-nthreads 1 -w /work
  docker run --rm -u $(id -u):$(id -g) -e MRIQC_DEV=1 \
      -v $PWD/testdata/dwi:/data:ro -v $PWD/out:/out -v $PWD/work:/work \
      nipreps/mriqc:ci /data /out group -m dwi --no-sub -w /work
  docker run --rm -v $PWD/out:/out:ro -v $PWD:/src:ro --entrypoint dfcheck \
      nipreps/mriqc:ci -i /out/group_dwi.tsv -r /src/mriqc/data/testdata/group_dwi.tsv

``MRIQC_DEV=1`` fixes the run identifier, so that the names of the log files match
the list of expected outputs.
