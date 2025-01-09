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
"""PyBIDS tooling."""

from __future__ import annotations

import json
import os
from pathlib import Path

DOI = 'https://doi.org/10.1371/journal.pone.0184661'


def write_bidsignore(deriv_dir):
    from mriqc.config import SUPPORTED_SUFFIXES

    bids_ignore = [
        '*.html',
        'logs/',  # Reports
    ] + [f'*_{suffix}.json' for suffix in SUPPORTED_SUFFIXES]

    ignore_file = Path(deriv_dir) / '.bidsignore'

    ignore_file.write_text('\n'.join(bids_ignore) + '\n')


def write_derivative_description(bids_dir, deriv_dir):
    from mriqc import __download__, __version__

    bids_dir = Path(bids_dir)
    deriv_dir = Path(deriv_dir)
    desc = {
        'BIDSVersion': '1.4.0',
        'DatasetType': 'derivative',
        'GeneratedBy': [
            {
                'Name': 'MRIQC',
                'Version': __version__,
                'CodeURL': __download__,
            }
        ],
        'HowToAcknowledge': f'Please cite our paper ({DOI}).',
    }

    # Keys that can only be set by environment
    # XXX: This currently has no effect, but is a stand-in to remind us to figure out
    # how to detect the container
    if 'MRIQC_DOCKER_TAG' in os.environ:
        desc['GeneratedBy'][0]['Container'] = {
            'Type': 'docker',
            'Tag': f'nipreps/mriqc:{os.environ["MRIQC_DOCKER_TAG"]}',
        }
    if 'MRIQC_SINGULARITY_URL' in os.environ:
        desc['GeneratedBy'][0]['Container'] = {
            'Type': 'singularity',
            'URI': os.getenv('MRIQC_SINGULARITY_URL'),
        }

    # Keys deriving from source dataset
    orig_desc = {}
    fname = bids_dir / 'dataset_description.json'
    if fname.exists():
        orig_desc = json.loads(fname.read_text())

    if 'Name' in orig_desc:
        desc['Name'] = f'MRIQC - {orig_desc["Name"]}'
    else:
        desc['Name'] = 'MRIQC - MRI Quality Control'

    if 'DatasetDOI' in orig_desc:
        desc['SourceDatasets'] = [
            {
                'URL': f'https://doi.org/{orig_desc["DatasetDOI"]}',
                'DOI': orig_desc['DatasetDOI'],
            }
        ]
    if 'License' in orig_desc:
        desc['License'] = orig_desc['License']

    Path.write_text(deriv_dir / 'dataset_description.json', json.dumps(desc, indent=4))


def derive_bids_fname(
    orig_path: str | Path,
    entity: str | None = None,
    newsuffix: str | None = None,
    newpath: str | Path | None = None,
    newext: str | None = None,
    position: int = -1,
    absolute: bool = True,
) -> Path | str:
    """
    Derive a new file name from a BIDS-formatted path.

    Parameters
    ----------
    orig_path : :obj:`str` or :obj:`os.pathlike`
        A filename (may or may not include path).
    entity : :obj:`str`, optional
        A new BIDS-like key-value pair.
    newsuffix : :obj:`str`, optional
        Replace the BIDS suffix.
    newpath : :obj:`str` or :obj:`os.pathlike`, optional
        Path to replace the path of the input orig_path.
    newext : :obj:`str`, optional
        Replace the extension of the file.
    position : :obj:`int`, optional
        Position to insert the entity in the filename.
    absolute : :obj:`bool`, optional
        If True (default), returns the absolute path of the modified filename.

    Returns
    -------
    Absolute path of the modified filename

    Examples
    --------
    >>> derive_bids_fname(
    ...     'sub-001/ses-01/anat/sub-001_ses-01_T1w.nii.gz',
    ...     entity='desc-preproc',
    ...     absolute=False,
    ... )
    PosixPath('sub-001/ses-01/anat/sub-001_ses-01_desc-preproc_T1w.nii.gz')

    >>> derive_bids_fname(
    ...     'sub-001/ses-01/anat/sub-001_ses-01_T1w.nii.gz',
    ...     entity='desc-brain',
    ...     newsuffix='mask',
    ...     newext=".nii",
    ...     absolute=False,
    ... )  # doctest: +ELLIPSIS
    PosixPath('sub-001/ses-01/anat/sub-001_ses-01_desc-brain_mask.nii')

    >>> derive_bids_fname(
    ...     'sub-001/ses-01/anat/sub-001_ses-01_T1w.nii.gz',
    ...     entity='desc-brain',
    ...     newsuffix='mask',
    ...     newext=".nii",
    ...     newpath="/output/node",
    ...     absolute=True,
    ... )  # doctest: +ELLIPSIS
    PosixPath('/output/node/sub-001_ses-01_desc-brain_mask.nii')

    >>> derive_bids_fname(
    ...     'sub-001/ses-01/anat/sub-001_ses-01_T1w.nii.gz',
    ...     entity='desc-brain',
    ...     newsuffix='mask',
    ...     newext=".nii",
    ...     newpath=".",
    ...     absolute=False,
    ... )  # doctest: +ELLIPSIS
    PosixPath('sub-001_ses-01_desc-brain_mask.nii')

    """

    orig_path = Path(orig_path)
    newpath = orig_path.parent if newpath is None else Path(newpath)

    ext = ''.join(orig_path.suffixes)
    newext = newext if newext is not None else ext
    orig_stem = orig_path.name.replace(ext, '')

    suffix = orig_stem.rsplit('_', maxsplit=1)[-1].strip('_')
    newsuffix = newsuffix.strip('_') if newsuffix is not None else suffix

    orig_stem = orig_stem.replace(suffix, '').strip('_')
    bidts = [bit for bit in orig_stem.split('_') if bit]
    if entity:
        if position == -1:
            bidts.append(entity)
        else:
            bidts.insert(position, entity.strip('_'))

    retval = newpath / f'{"_".join(bidts)}_{newsuffix}.{newext.strip(".")}'

    return retval.absolute() if absolute else retval
