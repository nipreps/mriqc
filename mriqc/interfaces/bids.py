#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# @Author: oesteban
# @Date:   2016-06-03 09:35:13
# @Last Modified by:   oesteban
# @Last Modified time: 2016-06-03 10:06:55
import os.path as op
import simplejson as json
from nipype.interfaces.base import (traits, isdefined, TraitedSpec, BaseInterface,
                                    BaseInterfaceInputSpec, File)


class ReadSidecarJSONInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True, desc='the input nifti file')
    fields = traits.List(traits.Str, desc='get only certain fields')

class ReadSidecarJSONOutputSpec(TraitedSpec):
    out_dict = traits.Dict()

class ReadSidecarJSON(BaseInterface):
    """
    An utility to find and read JSON sidecar files of a BIDS tree
    """

    input_spec = ReadSidecarJSONInputSpec
    output_spec = ReadSidecarJSONOutputSpec

    def __init__(self, **inputs):
        self._results = {}
        super(ReadSidecarJSON, self).__init__(**inputs)

    def _run_interface(self, runtime):
        metadata = get_metadata_for_nifti(self.inputs.in_file)

        if isdefined(self.inputs.fields) and self.inputs.fields:
            for fname in self.inputs.fields:
                self._results[fname] = metadata[fname]
        else:
            self._results = metadata

        return runtime

    def _list_outputs(self):
        out = self.output_spec().get()
        out['out_dict'] = self._results
        return out

def get_metadata_for_nifti(in_file):
    """Fetchs metadata for a given nifi file"""
    in_file = op.abspath(in_file)

    fname, ext = op.splitext(in_file)
    if ext == '.gz':
        fname, ext2 = op.splitext(fname)
        ext = ext2 + ext

    side_json = fname + '.json'
    fname_comps = side_json.split('/')[-1].split("_")

    session_comp_list = []
    subject_comp_list = []
    top_comp_list = []
    ses = None
    sub = None

    for comp in fname_comps:
        if comp[:3] != "run":
            session_comp_list.append(comp)
            if comp[:3] == "ses":
                ses = comp
            else:
                subject_comp_list.append(comp)
                if comp[:3] == "sub":
                    sub = comp
                else:
                    top_comp_list.append(comp)


    top_json = "/" + "_".join(top_comp_list)
    potential_json = [top_json]

    subject_json = "/" + sub + "/" + "_".join(subject_comp_list)
    potential_json.append(subject_json)

    if ses:
        session_json = "/" + sub + "/" + ses + "/" + "_".join(session_comp_list)
        potential_json.append(session_json)

    potential_json.append(side_json)

    merged_param_dict = {}
    for json_file_path in potential_json:
        if op.isfile(json_file_path):
            with open(json_file_path, 'r') as jsonfile:
                param_dict = json.load(jsonfile)
                merged_param_dict.update(param_dict)

    return merged_param_dict
