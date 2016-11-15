#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: oesteban
# @Date:   2015-11-19 16:44:27
# @Last Modified by:   oesteban
# @Last Modified time: 2016-11-14 11:03:27
""" MRIQC setup script """
from setuptools.command.build_py import build_py
from sys import version_info
PY3 = version_info[0] >= 3
PACKAGE_NAME = 'mriqc'

def main():
    """ Install entry-point """
    from os import path as op
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
        entry_points={'console_scripts': ['mriqc=mriqc.bin.mriqc_run:main',
                                          'mriqc_plot=mriqc.bin.mriqc_plot:main',
                                          'abide2bids=mriqc.bin.abide2bids:main',
                                          'fs2gif=mriqc.bin.fs2gif:main',
                                          'dfcheck=mriqc.bin.dfcheck:main',
                                          'participants=mriqc.bin.subject_wrangler:main']},
        packages=find_packages(exclude=['*.tests']),
        package_data={'mriqc': ['data/reports/*.rst',
                                'data/reports/*.html',
                                'data/reports/resources/*',
                                'data/tests/*']},
        zip_safe=False,
        cmdclass={'build_py': BuildWithCommitInfoCommand}
    )


class BuildWithCommitInfoCommand(build_py):
    """ Return extended build command class for recording commit
    The extended command tries to run git to find the current commit, getting
    the empty string if it fails.  It then writes the commit hash into a file
    in the `pkg_dir` path, named ``COMMIT_INFO.txt``.
    In due course this information can be used by the package after it is
    installed, to tell you what commit it was installed from if known.
    To make use of this system, you need a package with a COMMIT_INFO.txt file -
    e.g. ``myproject/COMMIT_INFO.txt`` - that might well look like this::
        # This is an ini file that may contain information about the code state
        [commit hash]
        # The line below may contain a valid hash if it has been substituted during 'git archive'
        archive_subst_hash=$Format:%h$
        # This line may be modified by the install process
        install_hash=
    The COMMIT_INFO file above is also designed to be used with git substitution
    - so you probably also want a ``.gitattributes`` file in the root directory
    of your working tree that contains something like this::
       myproject/COMMIT_INFO.txt export-subst
    That will cause the ``COMMIT_INFO.txt`` file to get filled in by ``git
    archive`` - useful in case someone makes such an archive - for example with
    via the github 'download source' button.
    Although all the above will work as is, you might consider having something
    like a ``get_info()`` function in your package to display the commit
    information at the terminal.  See the ``pkg_info.py`` module in the nipy
    package for an example.
    """
    def run(self):
        from io import open
        from os.path import join as pjoin

        build_py.run(self)

        version = _get_dev_version()

        if version:
            out_path = pjoin(self.build_lib, PACKAGE_NAME, 'VERSION')
            with open(out_path, 'wt' if PY3 else 'wb') as cfile:
                cfile.write(version)

def _get_dev_version():
    def _git(cmd):
        from subprocess import Popen, PIPE
        proc = Popen(cmd, stdout=PIPE, stderr=PIPE, shell=True)
        out_val, _ = proc.communicate()
        return out_val.decode() if PY3 else out_val

    return _git('git describe').strip()



if __name__ == '__main__':
    main()
