#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: oesteban
# @Date:   2015-11-19 16:44:27
# @Last Modified by:   oesteban
# @Last Modified time: 2016-05-12 17:46:31

"""
MRIQC Cross-validation

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import os.path as op
from argparse import ArgumentParser
from argparse import RawTextHelpFormatter

def main():
    """Entry point"""
    parser = ArgumentParser(description='MRI Quality Control',
                            formatter_class=RawTextHelpFormatter)

    g_input = parser.add_argument_group('Inputs')
    g_input.add_argument('-i', '--in_csv', action='store',
                         required=True)

    # g_outputs = parser.add_argument_group('Outputs')
    opts = parser.parse_args()

if __name__ == '__main__':
    main()
