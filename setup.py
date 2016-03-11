#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: oesteban
# @Date:   2015-11-19 16:44:27
# @Last Modified by:   oesteban
# @Last Modified time: 2016-03-11 13:49:49
""" MRIQC setup script """
import os
import sys

__version__ = '0.0.2rc1'


def main():
    """ Install entry-point """
    from glob import glob
    from setuptools import setup

    req_list = []
    with open('requirements.txt', 'r') as rfile:
        req_list = rfile.readlines()

    setup(
        name='mriqc',
        version=__version__,
        description='NR-IQMs (no-reference Image Quality Metrics) for MRI',
        author='oesteban',
        author_email='code@oscaresteban.es',
        maintainer_email='crn.poldracklab@gmail.com',
        url='http://mriqc.readthedocs.org/',
        download_url='https://pypi.python.org/packages/source/m/mriqc/mriqc-0.0.2rc1.tar.gz',
        license='3-clause BSD',
        entry_points={'console_scripts': ['mriqc=mriqc.run_mriqc:main',]},
        packages=['mriqc', 'mriqc.workflows', 'mriqc.interfaces', 'mriqc.reports', 'mriqc.utils'],
        package_data={'mriqc': ['reports/html/*.html', 'data/*.txt']},
        install_requires=req_list,
        zip_safe=False,
        classifiers=[
            'Development Status :: 3 - Alpha',
            'Intended Audience :: Science/Research',
            'Topic :: Scientific/Engineering :: Image Recognition',
            'License :: OSI Approved :: BSD License',
            'Programming Language :: Python :: 2.7',
        ],
    )

if __name__ == '__main__':
    local_path = os.path.dirname(os.path.abspath(sys.argv[0]))
    os.chdir(local_path)
    sys.path.insert(0, local_path)

    main()
