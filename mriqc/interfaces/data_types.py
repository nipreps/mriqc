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
ANALYZE_75 = """
      DT_NONE                    0
      DT_UNKNOWN                 0     / what it says, dude           /
      DT_BINARY                  1     / binary (1 bit/voxel)         /
      DT_UNSIGNED_CHAR           2     / unsigned char (8 bits/voxel) /
      DT_SIGNED_SHORT            4     / signed short (16 bits/voxel) /
      DT_SIGNED_INT              8     / signed int (32 bits/voxel)   /
      DT_FLOAT                  16     / float (32 bits/voxel)        /
      DT_COMPLEX                32     / complex (64 bits/voxel)      /
      DT_DOUBLE                 64     / double (64 bits/voxel)       /
      DT_RGB                   128     / RGB triple (24 bits/voxel)   /
      DT_ALL                   255     / not very useful (?)          /
"""
ADDED = """
      DT_UINT8                   2
      DT_INT16                   4
      DT_INT32                   8
      DT_FLOAT32                16
      DT_COMPLEX64              32
      DT_FLOAT64                64
      DT_RGB24                 128
"""
NEW_CODES = """
      DT_INT8                  256     / signed char (8 bits)         /
      DT_UINT16                512     / unsigned short (16 bits)     /
      DT_UINT32                768     / unsigned int (32 bits)       /
      DT_INT64                1024     / long long (64 bits)          /
      DT_UINT64               1280     / unsigned long long (64 bits) /
      DT_FLOAT128             1536     / long double (128 bits)       /
      DT_COMPLEX128           1792     / double pair (128 bits)       /
      DT_COMPLEX256           2048     / long double pair (256 bits)  /
      NIFTI_TYPE_UINT8           2 /! unsigned char. /
      NIFTI_TYPE_INT16           4 /! signed short. /
      NIFTI_TYPE_INT32           8 /! signed int. /
      NIFTI_TYPE_FLOAT32        16 /! 32 bit float. /
      NIFTI_TYPE_COMPLEX64      32 /! 64 bit complex = 2 32 bit floats. /
      NIFTI_TYPE_FLOAT64        64 /! 64 bit float = double. /
      NIFTI_TYPE_RGB24         128 /! 3 8 bit bytes. /
      NIFTI_TYPE_INT8          256 /! signed char. /
      NIFTI_TYPE_UINT16        512 /! unsigned short. /
      NIFTI_TYPE_UINT32        768 /! unsigned int. /
      NIFTI_TYPE_INT64        1024 /! signed long long. /
      NIFTI_TYPE_UINT64       1280 /! unsigned long long. /
      NIFTI_TYPE_FLOAT128     1536 /! 128 bit float = long double. /
      NIFTI_TYPE_COMPLEX128   1792 /! 128 bit complex = 2 64 bit floats. /
      NIFTI_TYPE_COMPLEX256   2048 /! 256 bit complex = 2 128 bit floats /
"""
