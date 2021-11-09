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
import re
from pathlib import Path

import simplejson as json
from mriqc import config
from mriqc.utils.misc import BIDS_COMP
from nipype.interfaces.base import (
    BaseInterfaceInputSpec,
    DynamicTraitedSpec,
    File,
    SimpleInterface,
    Str,
    TraitedSpec,
    Undefined,
    isdefined,
    traits,
)


class IQMFileSinkInputSpec(DynamicTraitedSpec, BaseInterfaceInputSpec):
    in_file = Str(mandatory=True, desc="path of input file")
    subject_id = Str(mandatory=True, desc="the subject id")
    modality = Str(mandatory=True, desc="the qc type")
    session_id = traits.Either(None, Str, usedefault=True)
    task_id = traits.Either(None, Str, usedefault=True)
    acq_id = traits.Either(None, Str, usedefault=True)
    rec_id = traits.Either(None, Str, usedefault=True)
    run_id = traits.Either(None, traits.Int, usedefault=True)
    dataset = Str(desc="dataset identifier")
    metadata = traits.Dict()
    provenance = traits.Dict()

    root = traits.Dict(desc="output root dictionary")
    out_dir = File(desc="the output directory")
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
    out_file = File(desc="the output JSON file containing the IQMs")


class IQMFileSink(SimpleInterface):
    input_spec = IQMFileSinkInputSpec
    output_spec = IQMFileSinkOutputSpec
    expr = re.compile("^root[0-9]+$")

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

        # Crawl back to the BIDS root
        path = Path(self.inputs.in_file)
        for i in range(1, 4):
            if str(path.parents[i].name).startswith("sub-"):
                bids_root = path.parents[i + 1]
                break
        in_file = str(path.relative_to(bids_root))

        # Build path and ensure directory exists
        bids_path = out_dir / in_file.replace("".join(Path(in_file).suffixes), ".json")
        bids_path.parent.mkdir(parents=True, exist_ok=True)
        self._results["out_file"] = str(bids_path)
        return self._results["out_file"]

    def _run_interface(self, runtime):
        out_file = self._gen_outfile()

        if isdefined(self.inputs.root):
            self._out_dict = self.inputs.root

        root_adds = []
        for key, val in list(self.inputs._outputs.items()):
            if not isdefined(val) or key == "trait_added":
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
                config.loggers.interface.warning(
                    'Output "%s" is not a dictionary (value="%s"), '
                    "discarding output.",
                    root_key,
                    str(val),
                )

        # Fill in the "bids_meta" key
        id_dict = {}
        for comp in list(BIDS_COMP.keys()):
            comp_val = getattr(self.inputs, comp, None)
            if isdefined(comp_val) and comp_val is not None:
                id_dict[comp] = comp_val
        id_dict["modality"] = self.inputs.modality

        if isdefined(self.inputs.metadata) and self.inputs.metadata:
            id_dict.update(self.inputs.metadata)

        if self._out_dict.get("bids_meta") is None:
            self._out_dict["bids_meta"] = {}
        self._out_dict["bids_meta"].update(id_dict)

        if isdefined(self.inputs.dataset):
            self._out_dict["bids_meta"]["dataset"] = self.inputs.dataset

        # Fill in the "provenance" key
        # Predict QA from IQMs and add to metadata
        prov_dict = {}
        if isdefined(self.inputs.provenance) and self.inputs.provenance:
            prov_dict.update(self.inputs.provenance)

        if self._out_dict.get("provenance") is None:
            self._out_dict["provenance"] = {}
        self._out_dict["provenance"].update(prov_dict)

        with open(out_file, "w") as f:
            f.write(
                json.dumps(
                    self._out_dict,
                    sort_keys=True,
                    indent=2,
                    ensure_ascii=False,
                )
            )

        return runtime


def _process_name(name, val):
    if "." in name:
        newkeys = name.split(".")
        name = newkeys.pop(0)
        nested_dict = {newkeys.pop(): val}

        for nk in reversed(newkeys):
            nested_dict = {nk: nested_dict}
        val = nested_dict

    return name, val
