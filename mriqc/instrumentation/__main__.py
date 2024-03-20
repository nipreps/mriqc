# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# Copyright 2022 The NiPreps Developers <nipreps@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# We support and encourage derived works from this project, please read
# about our expectations at
#
#     https://www.nipreps.org/community/licensing/
#

if __name__ == '__main__':
    import argparse

    from mriqc.instrumentation.resources import FindProcess, ResourceRecorder

    from . import __name__ as module

    # `python -m <module>` typically displays the command as __main__.py

    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name', type=str, required=True)
    parser.add_argument('-p', '--logfile_path', type=str, default='.')
    args = parser.parse_args()

    if '__main__.py' in argparse._sys.argv[0]:  # sys.argv[0]:
        argparse._sys.argv[0] = f'{argparse._sys.executable} -m {module}'

    pid = args.name if args.name.isnumeric() else FindProcess(args.name)

    ResourceRecorder(pid, log_file=args.logfile_path + str(pid) + '.tsv').run()
