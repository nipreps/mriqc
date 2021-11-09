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
import pandas as pd


def get_parser():
    """Entry point"""
    from argparse import ArgumentParser, RawTextHelpFormatter

    parser = ArgumentParser(
        description="Merge ratings from two raters",
        formatter_class=RawTextHelpFormatter,
    )
    g_input = parser.add_argument_group("Inputs")
    g_input.add_argument("rater_1", action="store")
    g_input.add_argument("rater_2", action="store")
    g_input.add_argument("--mapping-file", action="store")

    g_outputs = parser.add_argument_group("Outputs")
    g_outputs.add_argument("-o", "--output", action="store", default="merged.csv")
    return parser


def main():
    opts = get_parser().parse_args()

    rater_1 = pd.read_csv(opts.rater_1)[["participant_id", "check-1"]]
    rater_2 = pd.read_csv(opts.rater_2)[["participant_id", "check-1"]]

    rater_1.columns = ["participant_id", "rater_1"]
    rater_2.columns = ["participant_id", "rater_2"]
    merged = pd.merge(rater_1, rater_2, on="participant_id", how="outer")

    idcol = "participant_id"
    if opts.mapping_file:
        idcol = "subject_id"
        name_mapping = pd.read_csv(
            opts.mapping_file, sep=" ", header=None, usecols=[0, 1]
        )
        name_mapping.columns = ["subject_id", "participant_id"]
        name_mapping["participant_id"] = (
            name_mapping.participant_id.astype(str) + ".gif"
        )
        merged = pd.merge(name_mapping, merged, on="participant_id", how="outer")

    merged[[idcol, "rater_1", "rater_2"]].sort_values(by=idcol).to_csv(
        opts.output, index=False
    )


if __name__ == "__main__":
    main()
