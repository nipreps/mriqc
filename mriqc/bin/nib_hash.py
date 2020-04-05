#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: oesteban
# @Date:   2016-03-16 11:28:27
# @Last Modified by:   oesteban
# @Last Modified time: 2018-03-12 11:50:21

"""
Extracts the sha hash of the contents of a nifti file.

"""

import nibabel as nb
from hashlib import sha1


def get_parser():
    """ A trivial parser """
    from argparse import ArgumentParser, RawTextHelpFormatter

    parser = ArgumentParser(
        description="compare two pandas dataframes",
        formatter_class=RawTextHelpFormatter,
    )
    parser.add_argument("input_file", action="store", help="input nifti file")
    return parser


def get_hash(nii_file):
    """ Compute hash """
    data = nb.load(nii_file).get_data()
    data.flags.writeable = False
    return sha1(data.data.tobytes()).hexdigest()


def main():
    """Entry point"""
    fname = get_parser().parse_args().input_file
    sha = get_hash(fname)
    print("%s %s" % (sha, fname))


if __name__ == "__main__":
    main()
