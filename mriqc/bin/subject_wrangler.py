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
"""BIDS-Apps subject wrangler."""
import glob
import os.path as op
from argparse import ArgumentParser, RawTextHelpFormatter
from builtins import range  # pylint: disable=W0622
from random import shuffle
from textwrap import dedent

from mriqc import __version__
from mriqc.bin import messages

COMMAND = "{exec} {bids_dir} {out_dir} participant --participant_label {labels} {work_dir} {arguments} {logfile}"  # noqa: E501


def main():
    """Entry point."""
    parser = ArgumentParser(
        formatter_class=RawTextHelpFormatter,
        description=dedent(messages.SUBJECT_WRANGLER_DESCRIPTION),
    )

    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version=f"mriqc v{__version__}",
    )

    parser.add_argument(
        "bids_dir",
        action="store",
        help="The directory with the input dataset formatted according to the BIDS standard.",
    )
    parser.add_argument(
        "output_dir",
        action="store",
        help="The directory where the output files "
        "should be stored. If you are running group level analysis "
        "this folder should be prepopulated with the results of the"
        "participant level analysis.",
    )
    parser.add_argument(
        "--participant_label",
        "--subject_list",
        "-S",
        action="store",
        help="The label(s) of the participant(s) that should be analyzed. "
        "The label corresponds to sub-<participant_label> from the "
        'BIDS spec (so it does not include "sub-"). If this parameter '
        "is not provided all subjects should be analyzed. Multiple "
        "participants can be specified with a space separated list.",
        nargs="*",
    )
    parser.add_argument(
        "--group-size",
        default=1,
        action="store",
        type=int,
        help="Parallelize participants in groups.",
    )
    parser.add_argument(
        "--no-randomize",
        default=False,
        action="store_true",
        help="Do not randomize participants list before grouping.",
    )
    parser.add_argument(
        "--log-groups",
        default=False,
        action="store_true",
        help="Append logging output.",
    )
    parser.add_argument(
        "--multiple-workdir",
        default=False,
        action="store_true",
        help="Split work directories by jobs.",
    )
    parser.add_argument(
        "--bids-app-name",
        default="mriqc",
        action="store",
        help="BIDS app to call.",
    )
    parser.add_argument("--args", default="", action="store", help="Append arguments.")

    opts = parser.parse_args()

    # Build settings dict
    bids_dir = op.abspath(opts.bids_dir)
    subject_dirs = glob.glob(op.join(bids_dir, "sub-*"))
    all_subjects = sorted([op.basename(subj)[4:] for subj in subject_dirs])

    subject_list = opts.participant_label
    if subject_list is None or not subject_list:
        subject_list = all_subjects
    else:
        # remove sub- prefix, get unique
        for i, subj in enumerate(subject_list):
            subject_list[i] = subj[4:] if subj.startswith("sub-") else subj

        subject_list = sorted(list(set(subject_list)))

        if list(set(subject_list) - set(all_subjects)):
            non_exist = list(set(subject_list) - set(all_subjects))
            missing_label_error = messages.BIDS_LABEL_MISSING.format(
                label=" ".join(non_exist)
            )
            raise RuntimeError(missing_label_error)

    if not opts.no_randomize:
        shuffle(subject_list)

    gsize = opts.group_size

    if gsize < 0:
        raise RuntimeError(messages.BIDS_GROUP_SIZE)
    if gsize == 0:
        gsize = len(subject_list)

    j = i + gsize
    groups = [subject_list[i:j] for i in range(0, len(subject_list), gsize)]

    log_arg = ">> log/mriqc-{:04d}.log" if opts.log_groups else ""
    workdir_arg = " -w work/sjob-{:04d}" if opts.multiple_workdir else ""
    for i, part_group in enumerate(groups):
        kwargs = {
            "exec": opts.bids_app_name,
            "bids_dir": bids_dir,
            "out_dir": opts.output_dir,
            "labels": " ".join(part_group),
            "work_dir": workdir_arg.format(i),
            "arguments": opts.args,
            "logfile": log_arg.format(i),
        }
        print(COMMAND.format(**kwargs))


if __name__ == "__main__":
    main()
