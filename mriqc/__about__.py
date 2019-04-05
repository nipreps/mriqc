#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
MRIQC

"""

from datetime import date
from ._version import get_versions
__version__ = get_versions()['version']
del get_versions

__author__ = 'Oscar Esteban'
__email__ = 'code@oscaresteban.es'
__maintainer__ = 'Oscar Esteban'
__copyright__ = ('Copyright %d, Center for Reproducible Neuroscience, '
                 'Stanford University') % date.today().year
__credits__ = 'Oscar Esteban'
__license__ = '3-clause BSD'
__status__ = 'Prototype'
__description__ = """\
Automated Quality Control and visual reports for Quality Assesment of structural (T1w, T2w) \
and functional MRI of the brain\
"""
__longdesc__ = """\
MRIQC provides a series of image processing workflows \
to extract and compute a series of NR (no-reference), IQMs \
(image quality metrics) to be used in QAPs (quality \
assessment protocols) for MRI (magnetic \
resonance imaging). This open-source neuroimaging data \
processing tool is being developed as a part of the MRI \
image analysis and reproducibility platform offered by the \
CRN. This pipeline derives from, and is heavily influenced \
by, the PCP Quality Assessment Protocol. This tool extracts \
a series of IQMs from structural and functional MRI data. \
It is also scheduled to add diffusion MRI to the target \
imaging families.\
"""

__url__ = 'http://mriqc.readthedocs.org/'
__download__ = ('https://github.com/poldracklab/mriqc/archive/'
                '{}.tar.gz'.format(__version__))

PACKAGE_NAME = 'mriqc'
CLASSIFIERS = [
    'Development Status :: 3 - Alpha',
    'Intended Audience :: Science/Research',
    'Topic :: Scientific/Engineering :: Image Recognition',
    'License :: OSI Approved :: BSD License',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
]

SETUP_REQUIRES = []

REQUIRES = [
    'PyYAML',
    'dipy',
    'jinja2',
    'matplotlib',
    'nibabel>=2.2.1',
    'nilearn',
    'nipy',
    'nipype>=1.1.1',
    'nitime',
    'niworkflows<0.9.0a0,>=0.8.2',
    'numpy',
    'pandas>=0.21.0',
    'pybids<0.8.0a0,>=0.7.1',
    'scikit-image',
    'scikit-learn>=0.19.0',
    'scipy',
    'seaborn',
    'six',
    'statsmodels',
    'svgutils',
    'templateflow<0.2.0a0,>=0.1.7',
]

LINKS_REQUIRES = [
]


TESTS_REQUIRES = [
    'mock',
    'codecov',
    'pytest-xdist'
]

EXTRA_REQUIRES = {
    'doc': [
        'sphinx>=1.5.3',
        'sphinx_rtd_theme>=0.2.4',
        'sphinx-argparse',
        'pydotplus',
        'pydot>=1.2.3',
        'packaging',
    ],
    'tests': TESTS_REQUIRES,
    'duecredit': ['duecredit'],
    'notebooks': ['ipython', 'jupyter'],
    'classifier': ['xgboost']
}

# Enable a handle to install all extra dependencies at once
EXTRA_REQUIRES['all'] = [val for _, val in list(EXTRA_REQUIRES.items())]
