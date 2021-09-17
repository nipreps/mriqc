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
