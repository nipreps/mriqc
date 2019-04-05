#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# @Author: oesteban
# @Date:   2016-01-05 17:15:12
# @Email:  code@oscaresteban.es
# @Last modified by:   oesteban
"""Helper functions for the workflows"""
from distutils.version import StrictVersion
from builtins import range


def _tofloat(inlist):
    if isinstance(inlist, (list, tuple)):
        return [float(el) for el in inlist]
    else:
        return float(inlist)


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
    return {'fwhm_x': fwhm[0], 'fwhm_y': fwhm[1],
            'fwhm_z': fwhm[2], 'fwhm_avg': fwhm[3]}


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
        data, im.affine, im.header).to_filename(out_file)
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
    ftmask[size[0] // 2, size[1] // 2] = 0

    # Distance transform
    ftmask = distance(ftmask)
    ftmask /= ftmask.max()

    # Keep this just in case we want to switch to the opposite filter
    ftmask *= -1.0
    ftmask += 1.0

    ftmask[ftmask >= 0.4] = 1
    ftmask[ftmask < 1] = 0
    return ftmask


def slice_wise_fft(in_file, ftmask=None, spike_thres=3., out_prefix=None):
    """Search for spikes in slices using the 2D FFT"""
    import os.path as op
    import numpy as np
    import nibabel as nb
    from mriqc.workflows.utils import spectrum_mask
    from scipy.ndimage.filters import median_filter
    from scipy.ndimage import generate_binary_structure, binary_erosion
    from statsmodels.robust.scale import mad

    if out_prefix is None:
        fname, ext = op.splitext(op.basename(in_file))
        if ext == '.gz':
            fname, _ = op.splitext(fname)
        out_prefix = op.abspath(fname)

    func_data = nb.load(in_file).get_data()

    if ftmask is None:
        ftmask = spectrum_mask(tuple(func_data.shape[:2]))

    fft_data = []
    for t in range(func_data.shape[-1]):
        func_frame = func_data[..., t]
        fft_slices = []
        for z in range(func_frame.shape[2]):
            sl = func_frame[..., z]
            fftsl = median_filter(np.real(np.fft.fft2(sl)).astype(np.float32),
                                  size=(5, 5), mode='constant') * ftmask
            fft_slices.append(fftsl)
        fft_data.append(np.stack(fft_slices, axis=-1))

    # Recompose the 4D FFT timeseries
    fft_data = np.stack(fft_data, -1)

    # Z-score across t, using robust statistics
    mu = np.median(fft_data, axis=3)
    sigma = np.stack([mad(fft_data, axis=3)] * fft_data.shape[-1], -1)
    idxs = np.where(np.abs(sigma) > 1e-4)
    fft_zscored = fft_data - mu[..., np.newaxis]
    fft_zscored[idxs] /= sigma[idxs]

    # save fft z-scored
    out_fft = op.abspath(out_prefix + '_zsfft.nii.gz')
    nii = nb.Nifti1Image(fft_zscored.astype(np.float32), np.eye(4), None)
    nii.to_filename(out_fft)

    # Find peaks
    spikes_list = []
    for t in range(fft_zscored.shape[-1]):
        fft_frame = fft_zscored[..., t]

        for z in range(fft_frame.shape[-1]):
            sl = fft_frame[..., z]
            if np.all(sl < spike_thres):
                continue

            # Any zscore over spike_thres will be called a spike
            sl[sl <= spike_thres] = 0
            sl[sl > 0] = 1

            # Erode peaks and see how many survive
            struc = generate_binary_structure(2, 2)
            sl = binary_erosion(sl.astype(np.uint8), structure=struc).astype(np.uint8)

            if sl.sum() > 10:
                spikes_list.append((t, z))

    out_spikes = op.abspath(out_prefix + '_spikes.tsv')
    np.savetxt(out_spikes, spikes_list, fmt=b'%d', delimiter=b'\t', header='TR\tZ')

    return len(spikes_list), out_spikes, out_fft


def get_fwhmx():
    from nipype.interfaces.afni import Info, FWHMx
    fwhm_args = {"combine": True,
                 "detrend": True}
    afni_version = StrictVersion('%s.%s.%s' % Info.version())
    if afni_version >= StrictVersion("2017.2.3"):
        fwhm_args['args'] = '-ShowMeClassicFWHM'
    fwhm_interface = FWHMx(**fwhm_args)
    return fwhm_interface
