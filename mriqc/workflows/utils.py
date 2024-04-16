# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# Copyright 2021 The NiPreps Developers <nipreps@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# We support and encourage derived works from this project, please read
# about our expectations at
#
#     https://www.nipreps.org/community/licensing/
#
"""Helper functions for the workflows."""


def _tofloat(inlist):
    if isinstance(inlist, (list, tuple)):
        return [_tofloat(el) for el in inlist] if len(inlist) > 1 else _tofloat(inlist[0])
    return float(inlist)


def fwhm_dict(fwhm):
    """Convert a list of FWHM into a dictionary"""
    fwhm = [float(f) for f in fwhm]
    return {
        'fwhm_x': fwhm[0],
        'fwhm_y': fwhm[1],
        'fwhm_z': fwhm[2],
        'fwhm_avg': fwhm[3],
    }


def thresh_image(in_file, thres=0.5, out_file=None):
    """Thresholds an image"""
    import os.path as op

    import nibabel as nb
    import numpy as np

    if out_file is None:
        fname, ext = op.splitext(op.basename(in_file))
        if ext == '.gz':
            fname, ext2 = op.splitext(fname)
            ext = ext2 + ext
        out_file = op.abspath(f'{fname}_thresh{ext}')

    im = nb.load(in_file)
    data = np.asanyarray(im.dataobj)
    data[data < thres] = 0
    data[data > 0] = 1
    nb.Nifti1Image(data, im.affine, im.header).to_filename(out_file)
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


def slice_wise_fft(in_file, ftmask=None, spike_thres=3.0, out_prefix=None):
    """Search for spikes in slices using the 2D FFT"""
    import os.path as op

    import nibabel as nb
    import numpy as np
    from scipy.ndimage import binary_erosion, generate_binary_structure
    from scipy.ndimage.filters import median_filter
    from statsmodels.robust.scale import mad

    from mriqc.workflows.utils import spectrum_mask

    if out_prefix is None:
        fname, ext = op.splitext(op.basename(in_file))
        if ext == '.gz':
            fname, _ = op.splitext(fname)
        out_prefix = op.abspath(fname)

    func_data = nb.load(in_file).get_fdata()

    if ftmask is None:
        ftmask = spectrum_mask(tuple(func_data.shape[:2]))

    fft_data = []
    for t in range(func_data.shape[-1]):
        func_frame = func_data[..., t]
        fft_slices = []
        for z in range(func_frame.shape[2]):
            sl = func_frame[..., z]
            fftsl = (
                median_filter(
                    np.real(np.fft.fft2(sl)).astype(np.float32),
                    size=(5, 5),
                    mode='constant',
                )
                * ftmask
            )
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
            struct = generate_binary_structure(2, 2)
            sl = binary_erosion(sl.astype(np.uint8), structure=struct).astype(np.uint8)

            if sl.sum() > 10:
                spikes_list.append((t, z))

    out_spikes = op.abspath(out_prefix + '_spikes.tsv')
    np.savetxt(out_spikes, spikes_list, fmt=b'%d', delimiter=b'\t', header='TR\tZ')

    return len(spikes_list), out_spikes, out_fft


def get_fwhmx():
    from nipype.interfaces.afni import FWHMx, Info

    fwhm_args = {'combine': True, 'detrend': True}
    afni_version = Info.version()

    if afni_version and afni_version >= (2017, 2, 3):
        fwhm_args['args'] = '-ShowMeClassicFWHM'

    fwhm_interface = FWHMx(**fwhm_args)
    return fwhm_interface


def generate_filename(in_file, dirname=None, suffix='', extension=None):
    """
    Generate a nipype-like filename.

    >>> str(generate_filename("/path/to/input.nii.gz").relative_to(Path.cwd()))
    'input.nii.gz'

    >>> str(generate_filename(
    ...     "/path/to/input.nii.gz", dirname="/other/path",
    ... ))
    '/other/path/input.nii.gz'

    >>> str(generate_filename(
    ...     "/path/to/input.nii.gz", dirname="/other/path", extension="tsv",
    ... ))
    '/other/path/input.tsv'

    >>> str(generate_filename(
    ...     "/path/to/input.nii.gz", dirname="/other/path", extension=".tsv",
    ... ))
    '/other/path/input.tsv'

    >>> str(generate_filename(
    ...     "/path/to/input.nii.gz", dirname="/other/path", extension="",
    ... ))
    '/other/path/input'

    >>> str(generate_filename(
    ...     "/path/to/input.nii.gz", dirname="/other/path", extension="", suffix="_mod",
    ... ))
    '/other/path/input_mod'

    >>> str(generate_filename(
    ...     "/path/to/input.nii.gz", dirname="/other/path", extension="", suffix="mod",
    ... ))
    '/other/path/input_mod'

    >>> str(generate_filename(
    ...     "/path/to/input", dirname="/other/path", extension="tsv", suffix="mod",
    ... ))
    '/other/path/input_mod.tsv'

    """
    from pathlib import Path

    in_file = Path(in_file)
    in_ext = ''.join(in_file.suffixes)

    dirname = Path.cwd() if dirname is None else Path(dirname)

    if extension is not None:
        extension = extension if not extension or extension.startswith('.') else f'.{extension}'
    else:
        extension = in_ext

    stem = in_file.name[: -len(in_ext)] if in_ext else in_file.name

    if suffix and not suffix.startswith('_'):
        suffix = f'_{suffix}'

    return dirname / f'{stem}{suffix}{extension}'
