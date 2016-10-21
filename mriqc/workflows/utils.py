#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# @Author: oesteban
# @Date:   2016-01-05 17:15:12
# @Email:  code@oscaresteban.es
# @Last modified by:   oesteban
# @Last Modified time: 2016-10-10 18:45:19
"""Helper functions for the workflows"""
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from __future__ import unicode_literals
from builtins import range


def fmri_getidx(in_file, start_idx, stop_idx):
    """Heuristics to set the start and stop indices of fMRI series"""
    from nibabel import load
    from nipype.interfaces.base import isdefined
    nvols = load(in_file).shape[3]
    max_idx = nvols - 1

    if start_idx is None or not isdefined(start_idx) or start_idx < 0 or start_idx > max_idx:
        start_idx = 0

    if (stop_idx is None or not isdefined(stop_idx) or stop_idx < start_idx or
            stop_idx > max_idx):
        stop_idx = max_idx
    return start_idx, stop_idx


def fwhm_dict(fwhm):
    """Convert a list of FWHM into a dictionary"""
    fwhm = [float(f) for f in fwhm]
    return {'x': fwhm[0], 'y': fwhm[1],
            'z': fwhm[2], 'avg': fwhm[3]}


def fd_jenkinson(in_file, rmax=80., out_file=None):
    """
    Compute the :abbr:`FD (framewise displacement)` [Jenkinson2002]_
    on a 4D dataset, after AFNI-``3dvolreg`` has been executed
    (generally a file named ``*.affmat12.1D``).

    :param str in_file: path to epi file
    :param float rmax: the default radius (as in FSL) of a sphere represents
      the brain in which the angular displacements are projected.
    :param str out_file: a path for the output file with the FD

    :return: the output file with the FD, and the average FD along
      the time series
    :rtype: tuple(str, float)


    .. note ::

      :code:`infile` should have one 3dvolreg affine matrix in one row -
      NOT the motion parameters


    .. note :: Acknowledgments

      We thank Steve Giavasis (@sgiavasis) and Krishna Somandepali for their
      original implementation of this code in the [QAP]_.


    """

    import sys
    import math
    import os.path as op
    import numpy as np

    if out_file is None:
        fname, ext = op.splitext(op.basename(in_file))
        out_file = op.abspath('{}_fdfile{}'.format(fname, ext))

    pm_ = np.genfromtxt(in_file)
    original_shape = pm_.shape
    pm = np.zeros((pm_.shape[0], pm_.shape[1] + 4))
    pm[:, :original_shape[1]] = pm_
    pm[:, original_shape[1]:] = [0.0, 0.0, 0.0, 1.0]

    # rigid body transformation matrix
    T_rb_prev = np.matrix(np.eye(4))

    flag = 0
    X = [0]  # First timepoint
    for i in range(0, pm.shape[0]):
        # making use of the fact that the order of aff12 matrix is "row-by-row"
        T_rb = np.matrix(pm[i].reshape(4, 4))

        if flag == 0:
            flag = 1
        else:
            M = np.dot(T_rb, T_rb_prev.I) - np.eye(4)
            A = M[0:3, 0:3]
            b = M[0:3, 3]

            FD_J = math.sqrt(
                (rmax * rmax / 5) * np.trace(np.dot(A.T, A)) + np.dot(b.T, b))
            X.append(FD_J)

        T_rb_prev = T_rb
    np.savetxt(out_file, X)
    return out_file


def thresh_image(in_file, thres=0.5, out_file=None):
    """Thresholds an image"""
    import os.path as op
    import nibabel as nb

    if out_file is None:
        fname, ext = op.splitext(op.basename(in_file))
        if ext == '.gz':
            fname, ext2 = op.splitext(fname)
            ext = ext2 + ext
        out_file = op.abspath('{}_thresh{}'.format(fname, ext))

    im = nb.load(in_file)
    data = im.get_data()
    data[data < thres] = 0
    data[data > 0] = 1
    nb.Nifti1Image(
        data, im.get_affine(), im.get_header()).to_filename(out_file)
    return out_file


def spectrum_mask(size):
    """Creates a mask to filter the image of size size"""
    import numpy as np
    from scipy.ndimage.morphology import distance_transform_edt as distance

    ftmask = np.ones(size)

    # Set zeros on corners
    # ftmask[0, 0] = 0
    # ftmask[size[0] - 1, size[1] - 1] = 0
    # ftmask[0, size[1] - 1] = 0
    # ftmask[size[0] - 1, 0] = 0
    ftmask[size[0]/2, size[1]/2] = 0

    # Distance transform
    ftmask = distance(ftmask)
    ftmask /= ftmask.max()

    # Keep this just in case we want to switch to the opposite filter
    ftmask *= -1.0
    ftmask += 1.0

    ftmask[ftmask >= 0.4] = 1
    ftmask[ftmask < 1] = 0
    return ftmask
