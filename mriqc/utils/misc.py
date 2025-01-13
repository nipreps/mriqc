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
"""Helper functions."""

from __future__ import annotations

import asyncio
import json
from collections import OrderedDict
from collections.abc import Iterable
from functools import partial
from os import cpu_count
from pathlib import Path
from typing import Callable, TypeVar

import nibabel as nb
import numpy as np
import pandas as pd

try:
    from collections.abc import MutableMapping
except ImportError:
    from collections.abc import MutableMapping

R = TypeVar('R')

IMTYPES = {
    'T1w': 'anat',
    'T2w': 'anat',
    'bold': 'func',
    'dwi': 'dwi',
}

BIDS_COMP = OrderedDict(
    [
        ('subject_id', 'sub'),
        ('session_id', 'ses'),
        ('task_id', 'task'),
        ('acq_id', 'acq'),
        ('rec_id', 'rec'),
        ('run_id', 'run'),
    ]
)

BIDS_EXPR = """\
^sub-(?P<subject_id>[a-zA-Z0-9]+)(_ses-(?P<session_id>[a-zA-Z0-9]+))?\
(_task-(?P<task_id>[a-zA-Z0-9]+))?(_acq-(?P<acq_id>[a-zA-Z0-9]+))?\
(_rec-(?P<rec_id>[a-zA-Z0-9]+))?(_run-(?P<run_id>[a-zA-Z0-9]+))?\
"""


async def worker(job: Callable[[], R], semaphore) -> R:
    async with semaphore:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, job)


def reorder_csv(csv_file, out_file=None):
    """
    Put subject, session and scan in front of csv file

    :param str csv_file: the input csv file
    :param str out_file: if provided, a new csv file is created

    :return: the path to the file with the columns reordered


    """
    if isinstance(csv_file, list):
        csv_file = csv_file[-1]

    if out_file is None:
        out_file = csv_file

    dataframe = pd.read_csv(csv_file)
    cols = dataframe.columns.tolist()  # pylint: disable=no-member
    try:
        cols.remove('Unnamed: 0')
    except ValueError:
        # The column does not exist
        pass

    for col in ('scan', 'session', 'subject'):
        cols.remove(col)
        cols.insert(0, col)

    dataframe[cols].to_csv(out_file)
    return out_file


def rotate_files(fname):
    """A function to rotate file names"""
    import glob
    import os
    import os.path as op

    name, ext = op.splitext(fname)
    if ext == '.gz':
        name, ext2 = op.splitext(fname)
        ext = ext2 + ext

    if not op.isfile(fname):
        return

    prev = glob.glob(f'{name}.*{ext}')
    prev.insert(0, fname)
    prev.append(f'{name}.{len(prev) - 1:d}{ext}')
    for i in reversed(list(range(1, len(prev)))):
        os.rename(prev[i - 1], prev[i])


def bids_path(subid, sesid=None, runid=None, prefix=None, out_path=None, ext='json'):
    import os.path as op

    fname = f'{subid}'
    if prefix is not None:
        if not prefix.endswith('_'):
            prefix += '_'
        fname = prefix + fname
    if sesid is not None:
        fname += f'_ses-{sesid}'
    if runid is not None:
        fname += f'_run-{runid}'

    if out_path is not None:
        fname = op.join(out_path, fname)
    return op.abspath(fname + '.' + ext)


def generate_pred(derivatives_dir, output_dir, mod):
    """
    Reads the metadata in the JIQM (json iqm) files and
    generates a corresponding prediction CSV table
    """

    if mod != 'T1w':
        return None

    # If some were found, generate the CSV file and group report
    jsonfiles = list(output_dir.glob(f'sub-*/**/{IMTYPES[mod]}/sub-*_{mod}.json'))
    if not jsonfiles:
        return None

    headers = list(BIDS_COMP) + ['mriqc_pred']
    predictions = {k: [] for k in headers}

    for jsonfile in jsonfiles:
        with open(jsonfile) as jsondata:
            data = json.load(jsondata).pop('bids_meta', None)

        if data is None:
            continue

        for k in headers:
            predictions[k].append(data.pop(k, None))

    dataframe = pd.DataFrame(predictions).sort_values(by=list(BIDS_COMP))

    # Drop empty columns
    dataframe.dropna(axis='columns', how='all', inplace=True)

    bdits_cols = list(set(BIDS_COMP) & set(dataframe.columns.ravel()))

    # Drop duplicates
    dataframe.drop_duplicates(bdits_cols, keep='last', inplace=True)

    out_csv = Path(output_dir) / f'{mod}_predicted_qa_csv'
    dataframe[bdits_cols + ['mriqc_pred']].to_csv(str(out_csv), index=False)
    return out_csv


def generate_tsv(output_dir, mod):
    """
    Generates a tsv file from all json files in the derivatives directory
    """

    # If some were found, generate the CSV file and group report
    out_tsv = output_dir / (f'group_{mod}.tsv')
    jsonfiles = list(output_dir.glob(f'sub-*/**/{IMTYPES[mod]}/sub-*_{mod}.json'))
    if not jsonfiles:
        return None, out_tsv

    datalist = []
    for jsonfile in jsonfiles:
        dfentry = _read_and_save(jsonfile)

        if dfentry is not None:
            bids_name = str(Path(jsonfile.name).stem)
            dfentry.pop('bids_meta', None)
            dfentry.pop('provenance', None)
            dfentry['bids_name'] = bids_name
            datalist.append(dfentry)

    dataframe = pd.DataFrame(datalist)
    cols = dataframe.columns.tolist()  # pylint: disable=no-member
    dataframe = dataframe.sort_values(by=['bids_name'])

    # Drop duplicates
    dataframe.drop_duplicates(['bids_name'], keep='last', inplace=True)

    # Set filename at front
    cols.insert(0, cols.pop(cols.index('bids_name')))
    dataframe[cols].to_csv(str(out_tsv), index=False, sep='\t')
    return dataframe, out_tsv


def _read_and_save(in_file):
    return json.loads(Path(in_file).read_text()) or None


def _flatten(in_dict, parent_key='', sep='_'):
    items = []
    for k, val in list(in_dict.items()):
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(val, MutableMapping):
            items.extend(list(_flatten(val, new_key, sep=sep).items()))
        else:
            items.append((new_key, val))
    return dict(items)


def _flatten_dict(indict):
    out_qc = {}
    for k, value in list(indict.items()):
        if not isinstance(value, dict):
            out_qc[k] = value
        else:
            for subk, subval in list(value.items()):
                if not isinstance(subval, dict):
                    out_qc['_'.join([k, subk])] = subval
                else:
                    for ssubk, ssubval in list(subval.items()):
                        out_qc['_'.join([k, subk, ssubk])] = ssubval
    return out_qc


def _flatten_list(xs):
    for x in xs:
        if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
            yield from _flatten_list(x)
        else:
            yield x


def _datalad_get(input_list, nprocs=None):
    from mriqc import config

    if not config.execution.bids_dir_datalad or not config.execution.datalad_get:
        return

    # Delay datalad import until we're sure we'll need it
    import logging

    from datalad.api import get

    _dataladlog = logging.getLogger('datalad')
    _dataladlog.setLevel(logging.WARNING)

    config.loggers.cli.log(
        25, 'DataLad dataset identified, attempting to `datalad get` unavailable files.'
    )
    return get(
        list(_flatten_list(input_list)),
        dataset=str(config.execution.bids_dir),
        jobs=nprocs
        if not None
        else max(
            config.nipype.omp_nthreads,
            config.nipype.nprocs,
        ),
    )


def _file_meta_and_size(
    files: list | str,
    volmin: int | None = 1,
    volmax: int | None = None,
):
    """
    Identify the largest file size (allows multi-echo groups).

    Parameters
    ----------
    files : :obj:`list`
        List of :obj:`os.pathlike` or sublist of :obj:`os.pathlike` (multi-echo case)
        of files to be extracted.
    volmin : :obj:`int`
        Minimum number of volumes that inputs must have.
    volmax : :obj:`int`
        Maximum number of volumes that inputs must have.

    Returns
    -------
    :obj:`tuple`
        A tuple (metadata, entities, sizes, valid) of items containing the different
        aspects extracted from the input(s).

    """

    import os

    from mriqc import config

    multifile = isinstance(files, (list, tuple))
    if multifile:
        metadata = []
        _bids_list = []
        _size_list = []
        _valid_list = []

        for filename in files:
            metadata_i, entities_i, sizes_i, valid_i = _file_meta_and_size(
                filename,
                volmin=volmin,
                volmax=volmax,
            )

            # Add to output lists
            metadata.append(metadata_i)
            _bids_list.append(entities_i)
            _size_list.append(sizes_i)
            _valid_list.append(valid_i)

        valid = all(_valid_list) and len({_m['NumberOfVolumes'] for _m in metadata}) == 1
        return metadata, _merge_entities(_bids_list), np.sum(_size_list), valid

    metadata = config.execution.layout.get_metadata(files)
    entities = config.execution.layout.parse_file_entities(files)
    size = os.path.getsize(files) / (1024**3)

    metadata['FileSize'] = size
    metadata['FileSizeUnits'] = 'GB'

    try:
        nii = nb.load(files)
        nifti_len = nii.shape[3]
    except nb.filebasedimages.ImageFileError:
        nifti_len = None
    except IndexError:  # shape has only 3 elements
        nifti_len = 1 if nii.dataobj.ndim == 3 else -1

    valid = True
    if volmin is not None:
        valid = nifti_len >= volmin

    if valid and volmax is not None:
        valid = nifti_len <= volmax

    metadata['NumberOfVolumes'] = nifti_len

    return metadata, entities, size, valid


async def _extract_meta_and_size(
    filelist: list,
    volmin: int | None = 1,
    volmax: int | None = None,
    max_concurrent: int = min(cpu_count(), 12),
) -> tuple[list, list, list, list]:
    """
    Extract corresponding metadata and file size in GB.

    Parameters
    ----------
    filelist : :obj:`list`
        List of :obj:`os.pathlike` or sublist of :obj:`os.pathlike` (multi-echo case)
        of files to be extracted.
    volmin : :obj:`int`
        Minimum number of volumes that inputs must have.
    volmax : :obj:`int`
        Maximum number of volumes that inputs must have.
    max_concurrent : :obj:`int`
        Maximum number of concurrent coroutines (files or multi-echo sets).

    Returns
    -------
    :obj:`tuple`
        A tuple (metadata, entities, sizes, valid) of lists containing the different
        aspects extracted from inputs.

    """

    semaphore = asyncio.Semaphore(max_concurrent)
    tasks = []
    for filename in filelist:
        tasks.append(
            asyncio.create_task(
                worker(
                    partial(
                        _file_meta_and_size,
                        filename,
                        volmin=volmin,
                        volmax=volmax,
                    ),
                    semaphore,
                )
            )
        )

    # Gather guarantees the order of the output
    metadata, entities, sizes, valid = list(zip(*await asyncio.gather(*tasks)))
    return metadata, entities, sizes, valid


def initialize_meta_and_data(
    max_concurrent: int = min(cpu_count(), 12),
) -> None:
    """
    Mine data and metadata corresponding to the dataset.

    Get files if datalad enabled and extract the necessary metadata.

    Parameters
    ----------
    max_concurrent : :obj:`int`
        Maximum number of concurrent coroutines (files or multi-echo sets).

    Returns
    -------
    :obj:`None`

    """
    from mriqc import config

    # Datalad-get all files
    dataset = config.workflow.inputs.values()
    _datalad_get(dataset)

    # Extract metadata and filesize
    config.workflow.inputs_metadata = {}
    config.workflow.inputs_entities = {}
    config.workflow.biggest_file_gb = {}
    for mod, input_list in config.workflow.inputs.items():
        config.loggers.cli.log(
            25,
            f'Extracting metadata and entities for {len(input_list)} input runs '
            f"of modality '{mod}'...",
        )

        # Some modalities require a minimum number of volumes
        volmin = None
        if mod == 'bold':
            volmin = config.workflow.min_len_bold
        elif mod == 'dwi':
            volmin = config.workflow.min_len_dwi

        # Some modalities require a maximum number of volumes
        volmax = None
        if mod in ('T1w', 'T2w'):
            volmax = 1

        # Run extraction in a asyncio coroutine loop
        metadata, entities, size, valid = asyncio.run(
            _extract_meta_and_size(
                input_list,
                max_concurrent=max_concurrent,
                volmin=volmin,
                volmax=volmax,
            )
        )

        # Identify nonconformant files that need to be dropped (and drop them)
        if num_dropped := len(input_list) - np.sum(valid):
            config.loggers.workflow.warn(
                f'{num_dropped} cannot be processed (too short or too long)'
            )

            filtered_results = [
                _v[:-1]
                for _v in zip(input_list, metadata, entities, size, valid)
                if _v[-1] is True
            ]
            input_list, metadata, entities, size = list(zip(*filtered_results))
            config.workflow.inputs[mod] = input_list

        # Finalizing (write to config so that values are propagated)
        _max_size = np.max(size)
        config.workflow.inputs_metadata[mod] = metadata
        config.workflow.inputs_entities[mod] = entities
        config.workflow.biggest_file_gb[mod] = float(_max_size)  # Cast required to store YAML

        config.loggers.cli.log(
            25,
            f"File size ('{mod}'): {_max_size:.2f}|{np.mean(size):.2f} GB [maximum|average].",
        )


def _merge_entities(
    entities: list,
) -> dict:
    """
    Merge a list of dictionaries with entities dropping those with nonuniform values.

    Examples
    --------
    >>> _merge_entities([
    ...     {'subject': '001', 'session': '001'},
    ...     {'subject': '001', 'session': '002'},
    ... ])
    {'subject': '001'}

    >>> _merge_entities([
    ...     {'subject': '001', 'session': '002'},
    ...     {'subject': '001', 'session': '002'},
    ... ])
    {'subject': '001', 'session': '002'}

    >>> _merge_entities([
    ...     {'subject': '001', 'session': '002'},
    ...     {'subject': '001', 'session': '002', 'run': 1},
    ... ])
    {'subject': '001', 'session': '002'}

    >>> _merge_entities([
    ...     {'subject': '001', 'session': '002'},
    ...     {'subject': '001', 'run': 1},
    ... ])
    {'subject': '001'}

    """
    out_entities = {}

    bids_keys = set(entities[0].keys())
    for entities_i in entities[1:]:
        bids_keys.intersection_update(entities_i.keys())

    # Preserve ordering
    bids_keys = [_b for _b in entities[0].keys() if _b in bids_keys]

    for key in bids_keys:
        values = {_entities[key] for _entities in entities}
        if len(values) == 1:
            out_entities[key] = values.pop()

    return out_entities
