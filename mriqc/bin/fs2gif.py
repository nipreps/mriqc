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
Batch export freesurfer results to animated gifs.
"""
import os
import os.path as op
import subprocess as sp
from argparse import ArgumentParser, RawTextHelpFormatter
from errno import EEXIST
from shutil import rmtree
from tempfile import mkdtemp

import nibabel as nb
import numpy as np
from skimage import exposure


def main():
    """Entry point"""
    parser = ArgumentParser(
        description="Batch export freesurfer results to animated gifs.",
        formatter_class=RawTextHelpFormatter,
    )
    g_input = parser.add_argument_group("Inputs")
    g_input.add_argument("-s", "--subject-id", action="store")
    g_input.add_argument("-t", "--temp-dir", action="store")
    g_input.add_argument("--keep-temp", action="store_true", default=False)
    g_input.add_argument("--zoom", action="store_true", default=False)
    g_input.add_argument("--hist-eq", action="store_true", default=False)
    g_input.add_argument("--use-xvfb", action="store_true", default=False)

    g_outputs = parser.add_argument_group("Outputs")
    g_outputs.add_argument("-o", "--output-dir", action="store", default="fs2gif")

    opts = parser.parse_args()

    if opts.temp_dir is None:
        tmpdir = mkdtemp()
    else:
        tmpdir = op.abspath(opts.temp_dir)
        try:
            os.makedirs(tmpdir)
        except OSError as exc:
            if exc.errno != EEXIST:
                raise exc

    out_dir = op.abspath(opts.output_dir)
    try:
        os.makedirs(out_dir)
    except OSError as exc:
        if exc.errno != EEXIST:
            raise exc

    subjects_dir = os.getenv("SUBJECTS_DIR", op.abspath("subjects"))
    subject_list = [opts.subject_id]
    if opts.subject_id is None:
        subject_list = [
            op.basename(name)
            for name in os.listdir(subjects_dir)
            if op.isdir(os.path.join(subjects_dir, name))
        ]
    environ = os.environ.copy()
    environ["SUBJECTS_DIR"] = subjects_dir
    if opts.use_xvfb:
        environ["doublebufferflag"] = 1

    # tcl_file = pkgr.resource_filename('mriqc', 'data/fsexport.tcl')
    tcl_contents = """
SetOrientation 0
SetCursor 0 128 128 128
SetDisplayFlag 3 0
SetDisplayFlag 22 1
set i 0
"""

    for subid in subject_list:
        sub_path = op.join(subjects_dir, subid)
        tmp_sub = op.join(tmpdir, subid)
        try:
            os.makedirs(tmp_sub)
        except OSError as exc:
            if exc.errno != EEXIST:
                raise exc

        niifile = op.join(tmp_sub, "%s.nii.gz") % subid
        ref_file = op.join(sub_path, "mri", "T1.mgz")
        sp.call(
            ["mri_convert", op.join(sub_path, "mri", "norm.mgz"), niifile], cwd=tmp_sub
        )
        data = nb.load(niifile).get_data()
        data[data > 0] = 1

        # Compute brain bounding box
        indexes = np.argwhere(data)
        bbox_min = indexes.min(0)
        bbox_max = indexes.max(0) + 1
        center = np.average([bbox_min, bbox_max], axis=0)

        if opts.hist_eq:
            modnii = op.join(tmp_sub, "%s.nii.gz" % subid)
            ref_file = op.join(tmp_sub, "%s.mgz" % subid)
            img = nb.load(niifile)
            data = exposure.equalize_adapthist(img.get_data(), clip_limit=0.03)
            nb.Nifti1Image(data, img.affine, img.header).to_filename(modnii)
            sp.call(["mri_convert", modnii, ref_file], cwd=tmp_sub)

        if not opts.zoom:
            # Export tiffs for left hemisphere
            tcl_file = op.join(tmp_sub, "%s.tcl" % subid)
            with open(tcl_file, "w") as tclfp:
                tclfp.write(tcl_contents)
                tclfp.write(
                    "for { set slice %d } { $slice < %d } { incr slice } {"
                    % (bbox_min[2], bbox_max[2])
                )
                tclfp.write("    SetSlice $slice\n")
                tclfp.write("    RedrawScreen\n")
                tclfp.write(
                    '    SaveTIFF [format "%s/%s-' % (tmp_sub, subid)
                    + '%03d.tif" $i]\n'
                )
                tclfp.write("    incr i\n")
                tclfp.write("}\n")
                tclfp.write("QuitMedit\n")
            cmd = [
                "tkmedit",
                subid,
                "T1.mgz",
                "lh.pial",
                "-aux-surface",
                "rh.pial",
                "-tcl",
                tcl_file,
            ]
            if opts.use_xvfb:
                cmd = _xvfb_run() + cmd

            print("Running tkmedit: %s" % " ".join(cmd))
            sp.call(cmd, env=environ)
            # Convert to animated gif
            print("Stacking coronal slices")
            sp.call(
                [
                    "convert",
                    "-delay",
                    "10",
                    "-loop",
                    "0",
                    "%s/%s-*.tif" % (tmp_sub, subid),
                    "%s/%s.gif" % (out_dir, subid),
                ]
            )

        else:
            # Export tiffs for left hemisphere
            tcl_file = op.join(tmp_sub, "lh-%s.tcl" % subid)
            with open(tcl_file, "w") as tclfp:
                tclfp.write(tcl_contents)
                tclfp.write("SetZoomLevel 2")
                tclfp.write(
                    "for { set slice %d } { $slice < %d } { incr slice } {"
                    % (bbox_min[2], bbox_max[2])
                )
                tclfp.write(
                    "    SetZoomCenter %d %d $slice\n"
                    % (center[0] + 30, center[1] - 10)
                )
                tclfp.write("    SetSlice $slice\n")
                tclfp.write("    RedrawScreen\n")
                tclfp.write(
                    '    SaveTIFF [format "{}/{}-lh-%03d.tif" $i]\n'.format(
                        tmp_sub, subid
                    )
                )
                tclfp.write("    incr i\n")
                tclfp.write("}\n")
                tclfp.write("QuitMedit\n")
            cmd = ["tkmedit", subid, "norm.mgz", "lh.white", "-tcl", tcl_file]
            if opts.use_xvfb:
                cmd = _xvfb_run() + cmd

            print("Running tkmedit: %s" % " ".join(cmd))
            sp.call(cmd, env=environ)
            # Convert to animated gif
            print("Stacking coronal slices")

            # Export tiffs for right hemisphere
            tcl_file = op.join(tmp_sub, "rh-%s.tcl" % subid)
            with open(tcl_file, "w") as tclfp:
                tclfp.write(tcl_contents)
                tclfp.write("SetZoomLevel 2")
                tclfp.write(
                    "for { set slice %d } { $slice < %d } { incr slice } {"
                    % (bbox_min[2], bbox_max[2])
                )
                tclfp.write(
                    "    SetZoomCenter %d %d $slice\n"
                    % (center[0] - 30, center[1] - 10)
                )
                tclfp.write("    SetSlice $slice\n")
                tclfp.write("    RedrawScreen\n")
                tclfp.write(
                    '    SaveTIFF [format "{}/{}-rh-%03d.tif" $slice]\n'.format(
                        tmp_sub, subid
                    )
                )
                tclfp.write("    incr i\n")
                tclfp.write("}\n")
                tclfp.write("QuitMedit\n")
            cmd = ["tkmedit", subid, "norm.mgz", "rh.white", "-tcl", tcl_file]
            if opts.use_xvfb:
                cmd = _xvfb_run() + cmd

            print("Running tkmedit: %s" % " ".join(cmd))
            sp.call(cmd, env=environ)
            # Convert to animated gif
            print("Stacking coronal slices")
            sp.call(
                [
                    "convert",
                    "-delay",
                    "10",
                    "-loop",
                    "0",
                    "%s/%s-lh-*.tif" % (tmp_sub, subid),
                    "%s/%s-lh.gif" % (out_dir, subid),
                ]
            )
            sp.call(
                [
                    "convert",
                    "-delay",
                    "10",
                    "-loop",
                    "0",
                    "%s/%s-rh-*.tif" % (tmp_sub, subid),
                    "%s/%s-rh.gif" % (out_dir, subid),
                ]
            )

        if not opts.keep_temp:
            rmtree(tmp_sub, ignore_errors=True, onerror=_myerror)


def _xvfb_run(wait=5, server_args="-screen 0, 1600x1200x24", logs=None):
    """
    Wrap command with xvfb-run. Copied from:
    https://github.com/VUIIS/seam/blob/1dabd9ca5b1fc7d66ef7d41c34ea8d42d668a484/seam/util.py

    """
    if logs is None:
        logs = op.join(mkdtemp(), "fs2gif_xvfb")

    return [
        "xvfb-run",
        "-a",  # automatically get a free server number
        "-f {}.out".format(logs),
        "-e {}.err".format(logs),
        "--wait={:d}".format(wait),
        '--server-args="{}"'.format(server_args),
    ]


def _myerror(msg):
    print("WARNING: Error deleting temporal files: %s" % msg)


if __name__ == "__main__":
    main()
