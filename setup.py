#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: oesteban
# @Date:   2015-11-19 16:44:27
# @Last Modified by:   oesteban
<<<<<<< HEAD
# @Last Modified time: 2016-10-25 17:09:15
=======
# @Last Modified time: 2016-10-18 08:56:53
>>>>>>> master
""" MRIQC setup script """

import sys

PACKAGE_NAME = 'mriqc'

def main():
    """ Install entry-point """
    from os import path as op
    from glob import glob
    from inspect import getfile, currentframe
    from setuptools import setup, find_packages
    from io import open  # pylint: disable=W0622
    this_path = op.dirname(op.abspath(getfile(currentframe())))

    # Python 3: use a locals dictionary
    # http://stackoverflow.com/a/1463370/6820620
    ldict = locals()
    # Get version and release info, which is all stored in mriqc/info.py
    module_file = op.join(this_path, PACKAGE_NAME, 'info.py')
    with open(module_file) as infofile:
        pythoncode = [line for line in infofile.readlines() if not line.strip().startswith('#')]
        exec('\n'.join(pythoncode), globals(), ldict)  # pylint: disable=W0122

    setup(
        name=PACKAGE_NAME,
        version=ldict['__version__'],
        description=ldict['__description__'],
        long_description=ldict['__longdesc__'],
        author=ldict['__author__'],
        author_email=ldict['__email__'],
        license=ldict['__license__'],
        maintainer_email='crn.poldracklab@gmail.com',
        classifiers=ldict['CLASSIFIERS'],
        # Dependencies handling
        setup_requires=[],
        install_requires=ldict['REQUIRES'],
        dependency_links=ldict['LINKS_REQUIRES'],
        tests_require=ldict['TESTS_REQUIRES'],
        extras_require=ldict['EXTRA_REQUIRES'],
        url=ldict['URL'],
        download_url=ldict['DOWNLOAD_URL'],
        entry_points={'console_scripts': ['mriqc=mriqc.utils.mriqc_run:main',
                                          'mriqc_plot=mriqc.utils.mriqc_plot:main',
                                          'abide2bids=mriqc.utils.abide2bids:main',
                                          'fs2gif=mriqc.utils.fs2gif:main',
                                          'dfcheck=mriqc.utils.dfcheck:main',
                                          'participants=mriqc.utils.subject_wrangler:main',
                                          'mriqc_fit=mriqc.classifier.cv:main']},
        packages=find_packages(),
        package_data={'mriqc': ['data/reports/*.rst',
                                'data/reports/*.html',
                                'data/reports/reports/*',
                                'data/tests/*']},
        zip_safe=False,
    )

if __name__ == '__main__':
    main()
