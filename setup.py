#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: oesteban
# @Date:   2015-11-19 16:44:27
# @Last Modified by:   oesteban
""" MRIQC setup script """

def main():
    """ Install entry-point """
    import versioneer
    from setuptools import setup, find_packages
    from mriqc.__about__ import (
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
        __url__,
        __download__,
        PACKAGE_NAME,
        CLASSIFIERS,
        REQUIRES,
        LINKS_REQUIRES,
        TESTS_REQUIRES,
        EXTRA_REQUIRES,
    )

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
        url=__url__,
        download_url=__download__,
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
