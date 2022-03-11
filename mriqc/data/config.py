# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# Copyright 2021 The NiPreps Developers <nipreps@gmail.com>
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
"""Utilities: Jinja2 templates."""
from io import open  # pylint: disable=W0622

from pkg_resources import resource_filename as pkgrf


class Template(object):
    """
    Utility class for generating a config file from a jinja template.
    https://github.com/oesteban/endofday/blob/f2e79c625d648ef45b08cc1f11fd0bd84342d604/endofday/core/template.py
    """

    def __init__(self, template_str):
        import jinja2

        self.template_str = template_str
        self.env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(searchpath="/"),
            trim_blocks=True,
            lstrip_blocks=True,
        )

    def compile(self, configs):
        """Generates a string with the replacements"""
        template = self.env.get_template(self.template_str)
        return template.render(configs)

    def generate_conf(self, configs, path):
        """Saves the oucome after replacement on the template to file"""
        output = self.compile(configs)
        with open(path, "w+") as output_file:
            output_file.write(output)


class IndividualTemplate(Template):
    """Specific template for the individual report"""

    def __init__(self):
        super(IndividualTemplate, self).__init__(
            pkgrf("mriqc", "data/reports/individual.html")
        )


class GroupTemplate(Template):
    """Specific template for the individual report"""

    def __init__(self):
        super(GroupTemplate, self).__init__(pkgrf("mriqc", "data/reports/group.html"))
