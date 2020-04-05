#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: oesteban
# @Date:   2015-11-19 16:44:27


def get_parser():
    """Build parser object"""
    from argparse import ArgumentParser
    from argparse import RawTextHelpFormatter

    parser = ArgumentParser(
        description="MRIQCWebAPI: Check entries", formatter_class=RawTextHelpFormatter
    )
    parser.add_argument(
        "modality",
        action="store",
        choices=["T1w", "bold"],
        help="number of expected items in the database",
    )
    parser.add_argument(
        "expected",
        action="store",
        type=int,
        help="number of expected items in the database",
    )
    parser.add_argument(
        "--webapi-url",
        action="store",
        default="https://mriqc.nimh.nih.gov/api/v1/T1w",
        type=str,
        help="IP address where the MRIQC WebAPI is listening",
    )
    return parser


def main():
    """Entry point"""
    from requests import get
    import logging

    # Run parser
    MRIQC_LOG = logging.getLogger(__name__)
    opts = get_parser().parse_args()
    MRIQC_LOG.info("Sending GET to %s", opts.webapi_url)
    resp = get(opts.webapi_url).json()
    MRIQC_LOG.info("There are %d records in database", resp["_meta"]["total"])
    assert opts.expected == resp["_meta"]["total"]


if __name__ == "__main__":
    main()
