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
MRIQC Plot script.
"""
import os
import os.path as op
from argparse import ArgumentParser, RawTextHelpFormatter

from mriqc import __version__
from mriqc.bin import messages
from mriqc.reports import workflow_report


def main():
    """Entry point."""
    parser = ArgumentParser(
        description="MRI Quality Control.", formatter_class=RawTextHelpFormatter
    )

    g_input = parser.add_argument_group("Inputs")
    g_input.add_argument(
        "-d",
        "--data-type",
        action="store",
        nargs="*",
        choices=["anat", "func"],
        default=["anat", "func"],
    )
    g_input.add_argument(
        "-v",
        "--version",
        action="store_true",
        default=False,
        help="show current mriqc version",
    )

    g_input.add_argument(
        "--nthreads",
        action="store",
        default=0,
        type=int,
        help="number of threads",
    )

    g_outputs = parser.add_argument_group("Outputs")
    g_outputs.add_argument("-o", "--output-dir", action="store")
    g_outputs.add_argument(
        "-w",
        "--work-dir",
        action="store",
        default=op.join(os.getcwd(), "work"),
    )

    opts = parser.parse_args()

    if opts.version:
        version_message = messages.PLOT_REPORT_VERSION.format(version=__version__)
        print(version_message)
        exit(0)

    settings = {"output_dir": os.getcwd(), "nthreads": opts.nthreads}

    if opts.output_dir:
        settings["output_dir"] = op.abspath(opts.output_dir)

    if not op.exists(settings["output_dir"]):
        os.makedirs(settings["output_dir"])

    settings["work_dir"] = op.abspath(opts.work_dir)
    if not op.exists(settings["work_dir"]):
        raise RuntimeError(messages.PLOT_WORK_MISSING)

    for dtype in opts.data_type:
        workflow_report(dtype, settings)


if __name__ == "__main__":
    main()
