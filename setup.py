#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: oesteban
# @Date:   2015-11-19 16:44:27
# @Last Modified by:   oesteban
# @Last Modified time: 2016-01-18 20:50:16
""" MRIQC setup script """
import os
import sys

__version__ = '0.0.2rc1'


def main():
    """ Install entry-point """
    from glob import glob
    from setuptools import setup

    setup(
        name='mriqc',
        version=__version__,
        description='',
        author_email='crn.poldracklab@gmail.com',
        url='https://github.com/poldracklab/mriqc',
        download_url='',
        license='3-clause BSD',
        packages=['mriqc', 'mriqc.workflows', 'mriqc.interfaces', 'mriqc.reports'],
        package_data={'mriqc': ['reports/html/*.html']},
        scripts=glob("scripts/*.py"),
        install_requires=["nipype", "nibabel", "pandas", "seaborn", "pyPdf2",
                          "xhtml2pdf", "qap"],
        zip_safe=False)

if __name__ == "__main__":
    local_path = os.path.dirname(os.path.abspath(sys.argv[0]))
    os.chdir(local_path)
    sys.path.insert(0, local_path)

    main()
