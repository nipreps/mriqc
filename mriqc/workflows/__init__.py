#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
===================
The MRIQC workflows
===================

The anatomical workflow
-----------------------

.. workflow::

  import os.path as op
  from mriqc.workflows.anatomical import anat_qc_workflow
  datadir = op.abspath('data')
  wf = anat_qc_workflow([op.join(datadir, 'sub-001/anat/sub-001_T1w.nii.gz')],
                        settings={'bids_dir': datadir,
                                  'output_dir': op.abspath('out')})

"""
from __future__ import print_function, division, absolute_import, unicode_literals
from mriqc.workflows.anatomical import anat_qc_workflow
from mriqc.workflows.functional import fmri_qc_workflow
