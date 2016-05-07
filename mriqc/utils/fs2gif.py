#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: oesteban
# @Date:   2016-03-16 11:28:27
# @Last Modified by:   oesteban
# @Last Modified time: 2016-04-04 13:50:50

"""
Batch export freesurfer results to animated gifs

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import os.path as op
import subprocess as sp
from shutil import rmtree
from argparse import ArgumentParser
from argparse import RawTextHelpFormatter
from tempfile import mkdtemp
from errno import EEXIST
import glob
from six import string_types
import numpy as np
import nibabel as nb
from skimage import exposure

def main():
    """Entry point"""
    parser = ArgumentParser(description='Batch export freesurfer results to animated gifs',
                            formatter_class=RawTextHelpFormatter)
    g_input = parser.add_argument_group('Inputs')
    g_input.add_argument('-S', '--subjects-dir', action='store', default=os.getcwd())
    g_input.add_argument('-s', '--subject-id', action='store')
    g_input.add_argument('-t', '--temp-dir', action='store')
    g_input.add_argument('--keep-temp', action='store_true', default=False)
    g_input.add_argument('--zoom', action='store_true', default=False)
    g_input.add_argument('--hist-eq', action='store_true', default=False)
    g_outputs = parser.add_argument_group('Outputs')
    g_outputs.add_argument('-o', '--output-dir', action='store', default='fs2gif')

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

    subjects_dir = op.abspath(opts.subjects_dir)
    subject_list = opts.subject_id
    if subject_list is None:
        subject_list = [name for name in os.listdir(subjects_dir)
                        if op.isdir(os.path.join(subjects_dir, name))]
    elif isinstance(subject_list, string_types):
        if '*' not in subject_list:
            subject_list = [subject_list]
        else:
            all_dirs = [op.join(subjects_dir, name) for name in os.listdir(subjects_dir)
                        if op.isdir(os.path.join(subjects_dir, name))]
            pattern = glob.glob(op.abspath(op.join(subjects_dir, opts.subject_id)))
            subject_list = list(set(pattern).intersection(set(all_dirs)))

    environ = os.environ.copy()
    environ['SUBJECTS_DIR'] = subjects_dir
    # tcl_file = pkgr.resource_filename('mriqc', 'data/fsexport.tcl')
    tcl_contents = """
SetOrientation 0
SetCursor 0 128 128 128
SetDisplayFlag 3 0
SetDisplayFlag 22 1
set i 0
"""

    for sub_path in subject_list:
        subid = op.basename(sub_path)
        tmp_sub = op.join(tmpdir, subid)
        try:
            os.makedirs(tmp_sub)
        except OSError as exc:
            if exc.errno != EEXIST:
                raise exc

        niifile = op.join(tmp_sub, '%s.nii.gz') % subid
        ref_file = op.join(sub_path, 'mri', 'T1.mgz')
        sp.call(['mri_convert', op.join(sub_path, 'mri', 'norm.mgz'), niifile],
                cwd=tmp_sub)
        data = nb.load(niifile).get_data()
        data[data > 0] = 1

        # Compute brain bounding box
        indexes = np.argwhere(data)
        bbox_min = indexes.min(0)
        bbox_max = indexes.max(0) + 1
        center = np.average([bbox_min, bbox_max], axis=0)

        if opts.hist_eq:
            modnii = op.join(tmp_sub, '%s.nii.gz' % subid)
            ref_file = op.join(tmp_sub, '%s.mgz' % subid)
            img = nb.load(niifile)
            data = exposure.equalize_adapthist(img.get_data(), clip_limit=0.03)
            nb.Nifti1Image(data, img.get_affine(), img.get_header()).to_filename(modnii)
            sp.call(['mri_convert', modnii, ref_file], cwd=tmp_sub)


        if not opts.zoom:
            # Export tiffs for left hemisphere
            tcl_file = op.join(tmp_sub, '%s.tcl' % subid)
            with open(tcl_file, 'w') as tclfp:
                tclfp.write(tcl_contents)
                tclfp.write('for { set slice %d } { $slice < %d } { incr slice } {' % (bbox_min[2], bbox_max[2]))
                tclfp.write('    SetSlice $slice\n')
                tclfp.write('    RedrawScreen\n')
                tclfp.write('    SaveTIFF [format "%s/%s-' % (tmp_sub, subid) + '%03d.tif" $i]\n')
                tclfp.write('    incr i\n')
                tclfp.write('}\n')
                tclfp.write('QuitMedit\n')
            sp.call(['tkmedit', subid, 'T1.mgz', 'lh.pial', '-aux-surface', 'rh.pial', '-tcl', tcl_file], env=environ)
            # Convert to animated gif
            sp.call(['convert', '-delay', '10', '-loop', '0', '%s/%s-*.tif' % (tmp_sub, subid),
                     '%s/%s.gif' % (out_dir, subid)])

        else:
            # Export tiffs for left hemisphere
            tcl_file = op.join(tmp_sub, 'lh-%s.tcl' % subid)
            with open(tcl_file, 'w') as tclfp:
                tclfp.write(tcl_contents)
                tclfp.write('SetZoomLevel 2')
                tclfp.write('for { set slice %d } { $slice < %d } { incr slice } {' % (bbox_min[2], bbox_max[2]))
                tclfp.write('    SetZoomCenter %d %d $slice\n' % (center[0] + 30, center[1] - 10))
                tclfp.write('    SetSlice $slice\n')
                tclfp.write('    RedrawScreen\n')
                tclfp.write('    SaveTIFF [format "%s/%s-lh-' % (tmp_sub, subid) + '%03d.tif" $i]\n')
                tclfp.write('    incr i\n')
                tclfp.write('}\n')
                tclfp.write('QuitMedit\n')
            sp.call(['tkmedit', subid, 'norm.mgz', 'lh.white', '-tcl', tcl_file], env=environ)

            # Export tiffs for right hemisphere
            tcl_file = op.join(tmp_sub, 'rh-%s.tcl' % subid)
            with open(tcl_file, 'w') as tclfp:
                tclfp.write(tcl_contents)
                tclfp.write('SetZoomLevel 2')
                tclfp.write('for { set slice %d } { $slice < %d } { incr slice } {' % (bbox_min[2], bbox_max[2]))
                tclfp.write('    SetZoomCenter %d %d $slice\n' % (center[0] - 30, center[1] - 10))
                tclfp.write('    SetSlice $slice\n')
                tclfp.write('    RedrawScreen\n')
                tclfp.write('    SaveTIFF [format "%s/%s-rh-' % (tmp_sub, subid) + '%03d.tif" $slice]\n')
                tclfp.write('    incr i\n')
                tclfp.write('}\n')
                tclfp.write('QuitMedit\n')
            sp.call(['tkmedit', subid, 'norm.mgz', 'rh.white', '-tcl', tcl_file], env=environ)

            # Convert to animated gif
            sp.call(['convert', '-delay', '10', '-loop', '0', '%s/%s-lh-*.tif' % (tmp_sub, subid),
                     '%s/%s-lh.gif' % (out_dir, subid)])
            sp.call(['convert', '-delay', '10', '-loop', '0', '%s/%s-rh-*.tif' % (tmp_sub, subid),
                     '%s/%s-rh.gif' % (out_dir, subid)])


        if not opts.keep_temp:
            try:
                rmtree(tmp_sub)
            except:
                pass


if __name__ == '__main__':
    main()

