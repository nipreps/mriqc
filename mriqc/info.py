#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
MRIQC

"""
import sys

__versionbase__ = '0.8.7'
__versionrev__ = 'a0'
__version__ = __versionbase__ + __versionrev__
__author__ = 'Oscar Esteban'
__email__ = 'code@oscaresteban.es'
__maintainer__ = 'Oscar Esteban'
__copyright__ = ('Copyright 2016, Center for Reproducible Neuroscience, '
                 'Stanford University')
__credits__ = 'Oscar Esteban'
__license__ = '3-clause BSD'
__status__ = 'Prototype'
__description__ = 'NR-IQMs (no-reference Image Quality Metrics) for MRI'
__longdesc__ = """\
MRIQC provides a series of image processing workflows to extract and compute a series of \
NR (no-reference), IQMs (image quality metrics) to be used in QAPs (quality assessment \
protocols) for MRI (magnetic resonance imaging).
This open-source neuroimaging data processing tool is being developed as a part of the \
MRI image analysis and reproducibility platform offered by the CRN. This pipeline derives \
from, and is heavily influenced by, the PCP Quality Assessment Protocol.
This tool extracts a series of IQMs from structural and functional MRI data. It is also \
scheduled to add diffusion MRI to the target imaging families.
"""

URL = 'http://mriqc.readthedocs.org/'
DOWNLOAD_URL = ('https://pypi.python.org/packages/source/m/mriqc/'
                'mriqc-{}.tar.gz'.format(__version__))
CLASSIFIERS = [
    'Development Status :: 3 - Alpha',
    'Intended Audience :: Science/Research',
    'Topic :: Scientific/Engineering :: Image Recognition',
    'License :: OSI Approved :: BSD License',
    'Programming Language :: Python :: 2.7',
    'Programming Language :: Python :: 3.5',
]


REQUIRES = [
    'numpy',
    'future',
    'scipy',
    'lockfile',
    'six',
    'matplotlib',
    'nibabel',
    'niworkflows>=0.0.3a11',
    'pandas',
    'dipy',
    'jinja2',
    'seaborn',
    'pyPdf2',
    'PyYAML',
    'nitime',
    'nilearn',
    'sklearn',
    'scikit-learn',
    'nipype',
    'rst2pdf',
]

LINKS_REQUIRES = [
    'git+https://github.com/nipy/nipype.git#egg=nipype',
]

if sys.version_info[0] > 2:
    LINKS_REQUIRES += ['git+https://github.com/oesteban/rst2pdf.git@futurize/stage2#egg=rst2pdf',]

TESTS_REQUIRES = [
    'mock',
    'codecov',
    'pytest-xdist'
]

EXTRA_REQUIRES = {
    'doc': ['sphinx'],
    'tests': TESTS_REQUIRES,
    'duecredit': ['duecredit']
}

# Enable a handle to install all extra dependencies at once
EXTRA_REQUIRES['all'] = [val for _, val in list(EXTRA_REQUIRES.items())]
