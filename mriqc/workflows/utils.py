#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# @Author: oesteban
# @Date:   2016-01-05 17:15:12
# @Email:  code@oscaresteban.es
# @Last modified by:   oesteban
# @Last Modified time: 2016-05-04 15:15:14
"""Helper functions for the workflows"""

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
        out_file = op.abspath('%s_fdfile%s' % (fname, ext))

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
