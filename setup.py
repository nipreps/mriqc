#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: oesteban
# @Date:   2015-11-19 16:44:27
# @Last Modified by:   oesteban
""" MRIQC setup script """
from setuptools.command.build_py import build_py
from sys import version_info
import versioneer

PY3 = version_info[0] >= 3
PACKAGE_NAME = 'mriqc'

def main():
    """ Install entry-point """
    from os import path as op
    from inspect import getfile, currentframe
    from setuptools import setup, find_packages
    from io import open  # pylint: disable=W0622

    from mriqc.info import (
        __version__,
        __author__,
        __email__,
        __maintainer__,
        __copyright__,
        __credits__,
        __license__,
        __status__,
        __description__,
        __longdesc__,
        CLASSIFIERS,
        REQUIRES,
        LINKS_REQUIRES,
        TESTS_REQUIRES,
        EXTRA_REQUIRES,
        URL,
        DOWNLOAD_URL,
    )

    # this_path = op.dirname(op.abspath(getfile(currentframe())))

    # # Python 3: use a locals dictionary
    # # http://stackoverflow.com/a/1463370/6820620
    # ldict = locals()
    # # Get version and release info, which is all stored in mriqc/info.py
    # module_file = op.join(this_path, PACKAGE_NAME, 'info.py')
    # with open(module_file) as infofile:
    #     pythoncode = [line for line in infofile.readlines() if not line.strip().startswith('#')]
    #     exec('\n'.join(pythoncode), globals(), ldict)  # pylint: disable=W0122

    setup(
        name=PACKAGE_NAME,
        version=versioneer.get_version(),
        description=__description__,
        long_description=__longdesc__,
        author=__author__,
        author_email=__email__,
        license=__license__,
        maintainer_email='crn.poldracklab@gmail.com',
        classifiers=CLASSIFIERS,
        # Dependencies handling
        setup_requires=[],
        install_requires=REQUIRES,
        dependency_links=LINKS_REQUIRES,
        tests_require=TESTS_REQUIRES,
        extras_require=EXTRA_REQUIRES,
        url=URL,
        download_url=DOWNLOAD_URL,
        entry_points={'console_scripts': ['mriqc=mriqc.bin.mriqc_run:main',
                                          'mriqc_fit=mriqc.bin.mriqc_fit:main',
                                          'mriqc_clf=mriqc.bin.mriqc_clf:main',
                                          'mriqc_plot=mriqc.bin.mriqc_plot:main',
                                          'abide2bids=mriqc.bin.abide2bids:main',
                                          'fs2gif=mriqc.bin.fs2gif:main',
                                          'dfcheck=mriqc.bin.dfcheck:main',
                                          'nib-hash=mriqc.bin.nib_hash:main',
                                          'participants=mriqc.bin.subject_wrangler:main']},
        packages=find_packages(exclude=['*.tests']),
        package_data={'mriqc': ['data/*.yml',
                                'data/*.tfm',
                                'data/csv/*.csv',
                                'data/rfc-nzs-abide-1.0.pklz',
                                'data/rfc-nzs-full-1.0.pklz',
                                'data/reports/*.rst',
                                'data/reports/*.html',
                                'data/reports/resources/*',
                                'data/reports/embed_resources/*',
                                'data/tests/*',
                                'data/mni/*.nii.gz']},
        zip_safe=False,
        cmdclass=versioneer.get_cmdclass(),
    )


if __name__ == '__main__':
    main()
