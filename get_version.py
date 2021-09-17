#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @Author: oesteban
# @Date:   2017-06-13 09:42:38
import os.path as op
import sys


def main():
    from mriqc.__about__ import __version__

    print(__version__)


if __name__ == "__main__":
    sys.path.insert(0, op.abspath("."))
    main()
