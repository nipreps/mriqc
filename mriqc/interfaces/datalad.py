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
"""A :obj:`~nipype.interfaces.utility.base.IdentityInterface` with a grafted Datalad getter."""

from pathlib import Path
from nipype.interfaces.io import add_traits
from nipype.interfaces.base import (
    BaseInterface,
    Directory,
    DynamicTraitedSpec,
    File,
    isdefined,
)
from mriqc import config


class DataladIdentityInterface(BaseInterface):
    """Sneaks a ``datalad get`` in paths, if datalad is available."""

    _always_run = True

    input_spec = DynamicTraitedSpec

    def __init__(self, fields=None, dataset_path=None, mandatory_inputs=True, **inputs):
        super().__init__(**inputs)

        if not fields:
            raise ValueError("fields must be a non-empty list")

        add_traits(
            self.inputs,
            ["dataset_path"],
            trait_type=Directory(exists=True, mandatory=True),
        )

        # Each input must be in the fields.
        if set(inputs.keys()) - set(fields):
            raise ValueError(
                f"inputs not in the fields: {', '.join(set(inputs.keys()) - set(fields))}."
            )
        add_traits(
            self.inputs,
            fields,
            trait_type=File(nohash=True, mandatory=mandatory_inputs)
        )

        if dataset_path:
            inputs["dataset_path"] = dataset_path

        self.inputs.trait_set(**inputs)

    def _run_interface(self, runtime):
        inputs = self.inputs.get()
        dataset_path = inputs.pop("dataset_path")
        if (
            not isdefined(dataset_path)
            or not (Path(dataset_path) / ".datalad").exists()
        ):
            config.loggers.interface.info("Datalad interface without dataset path defined.")
            return runtime

        _dl_found = False
        try:
            from datalad.api import get

            _dl_found = True
        except ImportError:
            def get(*args, **kwargs):
                """Mock datalad get."""

        dataset_path = Path(dataset_path)
        for field, value in inputs.items():
            if not isdefined(value):
                continue

            _pth = Path(value)
            if not _pth.is_absolute():
                _pth = dataset_path / _pth

            _datalad_candidate = _pth.is_symlink() and not _pth.exists()
            if not _dl_found and _datalad_candidate:
                config.loggers.interface.warning("datalad was required but not found")
                return runtime

            if _datalad_candidate:
                try:
                    result = get(
                        _pth,
                        dataset=dataset_path
                    )
                except Exception as exc:
                    config.loggers.interface.warning(f"datalad get on {_pth} failed.")
                    if (
                        config.environment.exec_env == "docker"
                        and ("This repository is not initialized for use by git-annex, "
                             "but .git/annex/objects/ exists") in f"{exc}"
                    ):
                        config.loggers.interface.warning(
                            "Execution seems containerirzed with Docker, please make sure "
                            "you are not running as root. To do so, please add the argument "
                            "``-u $(id -u):$(id -g)`` to your command line."
                        )
                    else:
                        config.loggers.interface.warning(str(exc))
                else:
                    if result[0]["status"] == "error":
                        config.loggers.interface.warning(f"datalad get failed: {result}")

        return runtime

    def _outputs(self):
        return self.inputs

    def aggregate_outputs(self, runtime=None, needed_outputs=None):
        return self.inputs

    def _list_outputs(self):
        raise NotImplementedError
