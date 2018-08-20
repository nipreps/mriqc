#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: oesteban
# @Date:   2015-11-19 16:44:27
# @Last Modified by:   oesteban
""" MRIQC setup script """


def main():
    """ Install entry-point """
    import os
    from setuptools import setup, find_packages
    from mriqc.__about__ import (
        __author__,
        __email__,
        __license__,
        __description__,
        __longdesc__,
        __url__,
        __download__,
        PACKAGE_NAME,
        CLASSIFIERS,
        REQUIRES,
        SETUP_REQUIRES,
        LINKS_REQUIRES,
        TESTS_REQUIRES,
        EXTRA_REQUIRES,
    )

    pkg_data = {
        'mriqc': [
            'data/*.yml',
            'data/*.tfm',
            'data/csv/*.csv',
            'data/mclf_*.pklz',
            'data/reports/*.rst',
            'data/reports/*.html',
            'data/reports/resources/*',
            'data/reports/embed_resources/*',
            'data/tests/*',
            'data/mni/*.nii.gz',
        ],
    }

    version = None
    cmdclass = {}
    root_dir = os.path.dirname(os.path.realpath(__file__))
    if os.path.isfile(os.path.join(root_dir, PACKAGE_NAME, 'VERSION')):
        with open(os.path.join(root_dir, PACKAGE_NAME, 'VERSION')) as vfile:
            version = vfile.readline().strip()
        pkg_data[PACKAGE_NAME].insert(0, 'VERSION')

    if version is None:
        import versioneer
        version = versioneer.get_version()
        cmdclass = versioneer.get_cmdclass()

    setup(
        name=PACKAGE_NAME,
        version=version,
        description=__description__,
        long_description=__longdesc__,
        author=__author__,
        author_email=__email__,
        license=__license__,
        maintainer_email='crn.poldracklab@gmail.com',
        classifiers=CLASSIFIERS,
        # Dependencies handling
        setup_requires=SETUP_REQUIRES,
        install_requires=REQUIRES,
        dependency_links=LINKS_REQUIRES,
        tests_require=TESTS_REQUIRES,
        extras_require=EXTRA_REQUIRES,
        url=__url__,
        download_url=__download__,
        entry_points={'console_scripts': [
            'mriqc=mriqc.bin.mriqc_run:main',
            'mriqc_clf=mriqc.bin.mriqc_clf:main',
            'mriqc_plot=mriqc.bin.mriqc_plot:main',
            'abide2bids=mriqc.bin.abide2bids:main',
            'fs2gif=mriqc.bin.fs2gif:main',
            'dfcheck=mriqc.bin.dfcheck:main',
            'nib-hash=mriqc.bin.nib_hash:main',
            'participants=mriqc.bin.subject_wrangler:main',
            'mriqc_labeler=mriqc.bin.labeler:main',
            'mriqcwebapi_test=mriqc.bin.mriqcwebapi_test:main',
        ]},
        packages=find_packages(exclude=['*.tests']),
        package_data=pkg_data,
        zip_safe=False,
        cmdclass=cmdclass,
    )


if __name__ == '__main__':
    main()
