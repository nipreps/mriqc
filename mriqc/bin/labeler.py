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
import csv
import os
import random
import sys
import webbrowser

import numpy as np


def num_rows(data):
    for j in range(1, 4):
        if len(data[j]) == 0:
            return j
    return 4


def main():
    """read the input file"""
    print("Reading file sinfo.csv")
    csvfile = open("sinfo.csv", "rb")
    csvreader = csv.reader(csvfile)
    file = list(csvreader)

    # display statistics
    finished = [0.0, 0.0, 0.0]
    hold = np.zeros((3, len(file) - 1))
    hold[:] = np.nan
    total = 601
    for i in range(1, len(file)):
        for j in range(1, 4):
            if len(file[i][j]) > 0:
                finished[j - 1] = finished[j - 1] + 1
                hold[j - 1, i - 1] = int(file[i][j])
    finished = np.divide(np.round(np.divide(finished, total) * 1000), 10)
    print(f"Completed: {' '.join(['%g%%' % f for f in finished])}")
    print(f"Total: {np.round(np.divide(np.sum(finished), 3))}%")
    input("Waiting: [enter]")

    # file[1:] are all the rows
    order = range(1, len(file))
    random.shuffle(order)
    # pick a random row
    for row in order:
        # check how many entries it has
        current = num_rows(file[row])
        if current <= 1:
            # if less than 1, run the row
            print("Check participant #" + file[row][0])
            fname = os.getcwd() + "/abide/" + file[row][0]
            if os.path.isfile(fname):
                webbrowser.open("file://" + fname)
                quality = input("Quality? [-1/0/1/e/c] ")
                if quality == "e":
                    break
                if quality == "c":
                    print("Current comment: " + file[row][4])
                    comment = input("Comment: ")
                    if len(comment) > 0:
                        file[row][4] = comment
                    quality = input("Quality? [-1/0/1/e] ")
                if quality == "e":
                    break
                file[row][current] = quality
            else:
                print("File does not exist")

    print("Writing file sinfo.csv")
    outfile = open("sinfo.csv", "wb")
    csvwriter = csv.writer(outfile)
    csvwriter.writerows(file)
    print("Ending")


if __name__ == "__main__":
    main()
    sys.exit(0)
