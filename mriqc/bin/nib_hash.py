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
"""
Extracts the sha hash of the contents of a nifti file.
"""
from argparse import ArgumentParser, RawTextHelpFormatter
from hashlib import sha1

import nibabel as nb
from mriqc.bin import messages


def get_parser() -> ArgumentParser:
    """
    A trivial parser.

    Returns
    -------
    ArgumentParser
        nib_hash execution parser
    """

    parser = ArgumentParser(
        description="Compare two pandas dataframes.",
        formatter_class=RawTextHelpFormatter,
    )
    parser.add_argument("input_file", action="store", help="input nifti file")
    return parser


def get_hash(nii_file: str) -> str:
    """
    Computes the sha1 hash for a given NIfTI format file.

    Parameters
    ----------
    nii_file : str
        Path to *nii* file

    Returns
    -------
    str
        SHA1 hash
    """
    data = nb.load(nii_file).get_data()
    data.flags.writeable = False
    return sha1(data.data.tobytes()).hexdigest()


def main():
    """Entry point."""
    file_name = get_parser().parse_args().input_file
    sha = get_hash(file_name)
    message = messages.HASH_REPORT.format(sha=sha, file_name=file_name)
    print(message)


if __name__ == "__main__":
    main()
