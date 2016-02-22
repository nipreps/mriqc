#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: oesteban
# @Date:   2015-11-19 16:44:27
# @Last Modified by:   oesteban
# @Last Modified time: 2016-02-22 15:31:08
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
        entry_points={'console_scripts': ['mriqc=mriqc.run_mriqc:main',]},
        packages=['mriqc', 'mriqc.workflows', 'mriqc.interfaces', 'mriqc.reports', 'mriqc.utils'],
        package_data={'mriqc': ['reports/html/*.html']},
        install_requires=["nipype", "nibabel", "pandas", "seaborn", "pyPdf2",
                          "xhtml2pdf", "qap"],
        zip_safe=False,
        classifiers=[
            'Development Status :: 3 - Alpha',
            'Intended Audience :: MRI processing',
            'Topic :: 3D Image Processing :: Quality Assessment',
            'License :: OSI Approved :: 3-clause BSD License',
            'Programming Language :: Python :: 2.7',
        ],
    )

if __name__ == "__main__":
    local_path = os.path.dirname(os.path.abspath(sys.argv[0]))
    os.chdir(local_path)
    sys.path.insert(0, local_path)

    main()
