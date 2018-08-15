#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# @Author: oesteban
# @Date:   2016-06-03 09:35:13
from __future__ import print_function, division, absolute_import, unicode_literals
import os.path as op
from pathlib import Path
import re
import simplejson as json
from nipype import logging
from nipype.interfaces.base import (
    traits, isdefined, TraitedSpec, DynamicTraitedSpec, BaseInterfaceInputSpec,
    File, Undefined, Str, SimpleInterface
)
from ..utils.misc import BIDS_COMP, BIDS_EXPR

IFLOGGER = logging.getLogger('nipype.interface')


class ReadSidecarJSONInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True, desc='the input nifti file')
    fields = traits.List(Str, desc='get only certain fields')


class ReadSidecarJSONOutputSpec(TraitedSpec):
    subject_id = Str()
    session_id = Str()
    task_id = Str()
    acq_id = Str()
    rec_id = Str()
    run_id = Str()
    out_dict = traits.Dict()
    relative_path = Str()


class ReadSidecarJSON(SimpleInterface):
    """
    An utility to find and read JSON sidecar files of a BIDS tree
    """
    expr = re.compile(BIDS_EXPR)
    input_spec = ReadSidecarJSONInputSpec
    output_spec = ReadSidecarJSONOutputSpec

    def _run_interface(self, runtime):
        metadata = get_metadata_for_nifti(self.inputs.in_file)
        output_keys = [key for key in list(self.output_spec().get().keys()) if key.endswith('_id')]
        outputs = self.expr.search(op.basename(self.inputs.in_file)).groupdict()

        for key in output_keys:
            id_value = outputs.get(key)
            if id_value is not None:
                self._results[key] = outputs.get(key)

        if isdefined(self.inputs.fields) and self.inputs.fields:
            for fname in self.inputs.fields:
                self._results[fname] = metadata[fname]
        else:
            self._results['out_dict'] = metadata

        # Crawl back to the BIDS root
        path = Path(self.inputs.in_file)
        for i in range(1, 3):
            bids_root = path.parents[i]
            if str(bids_root).startswith('sub-'):
                break

        self._results['relative_path'] = str(path.relative_to(bids_root))
        return runtime


class IQMFileSinkInputSpec(DynamicTraitedSpec, BaseInterfaceInputSpec):
    in_file = Str(mandatory=True, desc='path of input file relative to BIDS root')
    subject_id = Str(mandatory=True, desc='the subject id')
    modality = Str(mandatory=True, desc='the qc type')
    session_id = traits.Either(None, Str, usedefault=True)
    task_id = traits.Either(None, Str, usedefault=True)
    acq_id = traits.Either(None, Str, usedefault=True)
    rec_id = traits.Either(None, Str, usedefault=True)
    run_id = traits.Either(None, Str, usedefault=True)
    metadata = traits.Dict()
    provenance = traits.Dict()

    root = traits.Dict(desc='output root dictionary')
    out_dir = File(desc='the output directory')
    _outputs = traits.Dict(value={}, usedefault=True)

    def __setattr__(self, key, value):
        if key not in self.copyable_trait_names():
            if not isdefined(value):
                super(IQMFileSinkInputSpec, self).__setattr__(key, value)
            self._outputs[key] = value
        else:
            if key in self._outputs:
                self._outputs[key] = value
            super(IQMFileSinkInputSpec, self).__setattr__(key, value)


class IQMFileSinkOutputSpec(TraitedSpec):
    out_file = File(desc='the output JSON file containing the IQMs')


class IQMFileSink(SimpleInterface):
    input_spec = IQMFileSinkInputSpec
    output_spec = IQMFileSinkOutputSpec
    expr = re.compile('^root[0-9]+$')

    def __init__(self, fields=None, force_run=True, **inputs):
        super(IQMFileSink, self).__init__(**inputs)

        if fields is None:
            fields = []

        self._out_dict = {}

        # Initialize fields
        fields = list(set(fields) - set(self.inputs.copyable_trait_names()))
        self._input_names = fields
        undefined_traits = {key: self._add_field(key) for key in fields}
        self.inputs.trait_set(trait_change_notify=False, **undefined_traits)

        if force_run:
            self._always_run = True

    def _add_field(self, name, value=Undefined):
        self.inputs.add_trait(name, traits.Any)
        self.inputs._outputs[name] = value
        return value

    def _gen_outfile(self):
        out_dir = Path()
        if isdefined(self.inputs.out_dir):
            out_dir = Path(self.inputs.out_dir)

        in_file = self.inputs.in_file
        bids_path = in_file.replace(''.join(Path(in_file).suffixes), '.json')
        self._results['out_file'] = str(out_dir / bids_path)
        return self._results['out_file']

    def _run_interface(self, runtime):
        out_file = self._gen_outfile()

        if isdefined(self.inputs.root):
            self._out_dict = self.inputs.root

        root_adds = []
        for key, val in list(self.inputs._outputs.items()):
            if not isdefined(val) or key == 'trait_added':
                continue

            if not self.expr.match(key) is None:
                root_adds.append(key)
                continue

            key, val = _process_name(key, val)
            self._out_dict[key] = val

        for root_key in root_adds:
            val = self.inputs._outputs.get(root_key, None)
            if isinstance(val, dict):
                self._out_dict.update(val)
            else:
                IFLOGGER.warn(
                    'Output "%s" is not a dictionary (value="%s"), '
                    'discarding output.', root_key, str(val))

        # Fill in the "bids_meta" key
        id_dict = {}
        for comp in list(BIDS_COMP.keys()):
            comp_val = getattr(self.inputs, comp, None)
            if isdefined(comp_val) and comp_val is not None:
                id_dict[comp] = comp_val
        id_dict['modality'] = self.inputs.modality

        if isdefined(self.inputs.metadata) and self.inputs.metadata:
            id_dict.update(self.inputs.metadata)

        if self._out_dict.get('bids_meta') is None:
            self._out_dict['bids_meta'] = {}
        self._out_dict['bids_meta'].update(id_dict)

        # Fill in the "provenance" key
        # Predict QA from IQMs and add to metadata
        prov_dict = {}
        if isdefined(self.inputs.provenance) and self.inputs.provenance:
            prov_dict.update(self.inputs.provenance)

        if self._out_dict.get('provenance') is None:
            self._out_dict['provenance'] = {}
        self._out_dict['provenance'].update(prov_dict)

        with open(out_file, 'w') as f:
            f.write(json.dumps(self._out_dict, sort_keys=True, indent=2,
                               ensure_ascii=False))

        return runtime


def get_metadata_for_nifti(in_file):
    """Fetchs metadata for a given nifi file"""
    in_file = op.abspath(in_file)

    fname, ext = op.splitext(in_file)
    if ext == '.gz':
        fname, ext2 = op.splitext(fname)
        ext = ext2 + ext

    side_json = fname + '.json'
    fname_comps = op.basename(side_json).split("_")

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

    if any([comp.startswith('ses') for comp in fname_comps]):
        bids_dir = '/'.join(op.dirname(in_file).split('/')[:-3])
    else:
        bids_dir = '/'.join(op.dirname(in_file).split('/')[:-2])

    top_json = op.join(bids_dir, "_".join(top_comp_list))
    potential_json = [top_json]

    subject_json = op.join(bids_dir, sub, "_".join(subject_comp_list))
    potential_json.append(subject_json)

    if ses:
        session_json = op.join(bids_dir, sub, ses, "_".join(session_comp_list))
        potential_json.append(session_json)

    potential_json.append(side_json)

    merged_param_dict = {}
    for json_file_path in potential_json:
        if op.isfile(json_file_path):
            with open(json_file_path, 'r') as jsonfile:
                param_dict = json.load(jsonfile)
                merged_param_dict.update(param_dict)

    return merged_param_dict


def _process_name(name, val):
    if '.' in name:
        newkeys = name.split('.')
        name = newkeys.pop(0)
        nested_dict = {newkeys.pop(): val}

        for nk in reversed(newkeys):
            nested_dict = {nk: nested_dict}
        val = nested_dict

    return name, val
