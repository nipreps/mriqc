#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
MRIQC

"""

__versionbase__ = '0.9.0'
__versionrev__ = '-0'
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
    'numpy>=1.12.0',
    'nipype>=0.13.0rc1',
    'niworkflows>=0.0.5',
    'future',
    'scipy',
    'six',
    'matplotlib',
    'nibabel',
    'pandas',
    'dipy',
    'jinja2',
    'seaborn',
    'PyYAML',
    'nitime',
    'nilearn',
    'svgutils',
    'nipy',
    'statsmodels',
    'pybids'
]

LINKS_REQUIRES = []

TESTS_REQUIRES = [
    'mock',
    'codecov',
    'pytest-xdist'
]

EXTRA_REQUIRES = {
    'doc': ['sphinx>=1.4'],
    'tests': TESTS_REQUIRES,
    'duecredit': ['duecredit'],
    'notebooks': ['ipython', 'jupyter'],
    'classifier': ['scikit-learn', 'sklearn']
}

# Enable a handle to install all extra dependencies at once
EXTRA_REQUIRES['all'] = [val for _, val in list(EXTRA_REQUIRES.items())]
