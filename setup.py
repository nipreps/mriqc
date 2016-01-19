#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: oesteban
# @Date:   2015-11-19 16:44:27
# @Last Modified by:   oesteban
# @Last Modified time: 2016-01-18 20:50:16


def main():
    from glob import glob
    from setuptools import setup

    exec(open('mriqc/version.py').read())
    setup(
        name='mriqc',
        version=__version__,
        description='',
        author_email='crn.poldracklab@gmail.com',
        url='https://github.com/poldracklab/mriqc',
        download_url='',
        license='3-clause BSD',
        packages=['mriqc', 'mriqc.workflows', 'mriqc.interfaces'],
        package_data={'mriqc': ['html/*.html']},
        scripts=glob("scripts/*.py"),
        install_requires=["nipype", "nibabel", "pandas", "seaborn", "pyPdf2",
                          "xhtml2pdf", "qap"],
        zip_safe=False)

if __name__ == "__main__":
    import os
    import sys

    local_path = os.path.dirname(os.path.abspath(sys.argv[0]))
    os.chdir(local_path)
    sys.path.insert(0, local_path)

    main()
