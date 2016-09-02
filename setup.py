#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: oesteban
# @Date:   2015-11-19 16:44:27
# @Last Modified by:   oesteban
# @Last Modified time: 2016-08-29 18:40:10
""" MRIQC setup script """
import os
import sys

PACKAGE_NAME = 'mriqc'

def main():
    """ Install entry-point """
    from glob import glob
    from inspect import getfile, currentframe
    from setuptools import setup, find_packages

    this_path = os.path.dirname(os.path.abspath(getfile(currentframe())))
    # Read vars from info file
    module_file =os.path.join(this_path, PACKAGE_NAME, 'info.py')
    with open(module_file) as fvars:
        exec(compile(fvars.read(), module_file, 'exec'))


    REQ_LINKS = []
    with open('requirements.txt', 'r') as rfile:
        REQUIREMENTS = [line.strip() for line in rfile.readlines()]

    for i, req in enumerate(REQUIREMENTS):
        if req.startswith('-e'):
            REQUIREMENTS[i] = req.split('=')[1]
            REQ_LINKS.append(req.split()[1])

    if REQUIREMENTS is None:
        REQUIREMENTS = []

    setup(
        name=PACKAGE_NAME,
        version=__version__,
        description=__description__,
        long_description=__longdesc__,
        author=__author__,
        author_email=__email__,
        license=__license__,
        maintainer_email='crn.poldracklab@gmail.com',
        url='http://mriqc.readthedocs.org/',
        download_url='https://pypi.python.org/packages/source/m/mriqc/'
                     'mriqc-%s.tar.gz' % __version__,
        entry_points={'console_scripts': ['mriqc=mriqc.utils.mriqc_run:main',
                                          'mriqc_plot=mriqc.utils.mriqc_plot:main',
                                          'abide2bids=mriqc.utils.abide2bids:main',
                                          'fs2gif=mriqc.utils.fs2gif:main',
                                          'dfcheck=mriqc.utils.dfcheck:main',
                                          'participants=mriqc.utils.subject_wrangler:main']},
        install_requires=REQUIREMENTS,
        dependency_links=REQ_LINKS,
        packages=find_packages(),
        package_data={'mriqc': ['data/t1-mni_registration.json',
                                'data/t1-mni_registration_testing.json',
                                'data/reports/*.rst',
                                'data/tests/*']},
        zip_safe=False,
        classifiers=[
            'Development Status :: 3 - Alpha',
            'Intended Audience :: Science/Research',
            'Topic :: Scientific/Engineering :: Image Recognition',
            'License :: OSI Approved :: BSD License',
            'Programming Language :: Python :: 2.7',
            'Programming Language :: Python :: 3.5',
        ],
    )

if __name__ == '__main__':
    LOCAL_PATH = os.path.dirname(os.path.abspath(sys.argv[0]))
    os.chdir(LOCAL_PATH)
    sys.path.insert(0, LOCAL_PATH)

    main()
