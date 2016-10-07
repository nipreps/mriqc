#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

from __future__ import print_function, division, absolute_import, unicode_literals

from nipype.interfaces.base import BaseInterface


class MRIQCBaseInterface(BaseInterface):
    """
    Adds the _results property and implements _list_outputs

    """

    def __init__(self, **inputs):
        self._results = {}
        super(MRIQCBaseInterface, self).__init__(**inputs)

    def _list_outputs(self):
        return self._results
