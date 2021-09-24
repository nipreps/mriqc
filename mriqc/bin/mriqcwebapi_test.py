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
from mriqc.bin import messages


def get_parser():
    """
    Build parser object.
    """
    from argparse import ArgumentParser, RawTextHelpFormatter

    parser = ArgumentParser(
        description="MRIQCWebAPI: Check entries.", formatter_class=RawTextHelpFormatter
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
    """Entry point."""
    import logging

    from requests import get

    # Run parser
    MRIQC_LOG = logging.getLogger(__name__)
    opts = get_parser().parse_args()
    get_log_message = messages.WEBAPI_GET.format(address=opts.webapi_url)
    MRIQC_LOG.info(get_log_message)
    response = get(opts.webapi_url).json()
    n_records = response["_meta"]["total"]
    response_log_message = messages.WEBAPI_REPORT.format(n_records=n_records)
    MRIQC_LOG.info(response_log_message)
    assert opts.expected == n_records


if __name__ == "__main__":
    main()
