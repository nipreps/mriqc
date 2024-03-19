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
from nipype.interfaces.base import (
    BaseInterfaceInputSpec,
    Bunch,
    File,
    SimpleInterface,
    Str,
    TraitedSpec,
    isdefined,
    traits,
)

from mriqc import config, messages

# metadata whitelist
META_WHITELIST = [
    'AccelNumReferenceLines',
    'AccelerationFactorPE',
    'AcquisitionMatrix',
    'CogAtlasID',
    'CogPOID',
    'CoilCombinationMethod',
    'ContrastBolusIngredient',
    'ConversionSoftware',
    'ConversionSoftwareVersion',
    'DelayTime',
    'DeviceSerialNumber',
    'EchoTime',
    'EchoTrainLength',
    'EffectiveEchoSpacing',
    'FlipAngle',
    'GradientSetType',
    'HardcopyDeviceSoftwareVersion',
    'ImageType',
    'ImagingFrequency',
    'InPlanePhaseEncodingDirection',
    'InstitutionAddress',
    'InstitutionName',
    'Instructions',
    'InversionTime',
    'MRAcquisitionType',
    'MRTransmitCoilSequence',
    'MagneticFieldStrength',
    'Manufacturer',
    'ManufacturersModelName',
    'MatrixCoilMode',
    'MultibandAccelerationFactor',
    'NumberOfAverages',
    'NumberOfPhaseEncodingSteps',
    'NumberOfVolumesDiscardedByScanner',
    'NumberOfVolumesDiscardedByUser',
    'NumberShots',
    'ParallelAcquisitionTechnique',
    'ParallelReductionFactorInPlane',
    'PartialFourier',
    'PartialFourierDirection',
    'PatientPosition',
    'PercentPhaseFieldOfView',
    'PercentSampling',
    'PhaseEncodingDirection',
    'PixelBandwidth',
    'ProtocolName',
    'PulseSequenceDetails',
    'PulseSequenceType',
    'ReceiveCoilName',
    'RepetitionTime',
    'ScanOptions',
    'ScanningSequence',
    'SequenceName',
    'SequenceVariant',
    'SliceEncodingDirection',
    'SoftwareVersions',
    'TaskDescription',
    'TaskName',
    'TotalReadoutTime',
    'TotalScanTimeSec',
    'TransmitCoilName',
    'VariableFlipAngleFlag',
    'acq_id',
    'modality',
    'run_id',
    'subject_id',
    'task_id',
    'session_id',
]

PROV_WHITELIST = ['version', 'md5sum', 'software', 'settings']

HASH_BIDS = ['subject_id', 'session_id']


class UploadIQMsInputSpec(BaseInterfaceInputSpec):
    in_iqms = File(exists=True, mandatory=True, desc='the input IQMs-JSON file')
    endpoint = Str(mandatory=True, desc='URL of the POST endpoint')
    auth_token = Str(mandatory=True, desc='authentication token')
    email = Str(desc='set sender email')
    strict = traits.Bool(
        False, usedefault=True, desc='crash if upload was not successful'
    )


class UploadIQMsOutputSpec(TraitedSpec):
    api_id = traits.Either(
        None, traits.Str, desc='Id for report returned by the web api'
    )


class UploadIQMs(SimpleInterface):
    """
    Upload features to MRIQCWebAPI
    """

    input_spec = UploadIQMsInputSpec
    output_spec = UploadIQMsOutputSpec
    always_run = True

    def _run_interface(self, runtime):
        email = None
        if isdefined(self.inputs.email):
            email = self.inputs.email

        self._results['api_id'] = None

        response = upload_qc_metrics(
            self.inputs.in_iqms,
            endpoint=self.inputs.endpoint,
            auth_token=self.inputs.auth_token,
            email=email,
        )

        try:
            self._results['api_id'] = response.json()['_id']
        except (AttributeError, KeyError, ValueError):
            # response did not give us an ID
            errmsg = (
                'QC metrics upload failed to create an ID for the record '
                f'uplOADED. rEsponse from server follows: {response.text}'
            )
            config.loggers.interface.warning(errmsg)

        if response.status_code == 201:
            config.loggers.interface.info(messages.QC_UPLOAD_COMPLETE)
            return runtime

        errmsg = 'QC metrics failed to upload. Status %d: %s' % (
            response.status_code,
            response.text,
        )
        config.loggers.interface.warning(errmsg)
        if self.inputs.strict:
            raise RuntimeError(response.text)

        return runtime


def upload_qc_metrics(
    in_iqms,
    endpoint=None,
    email=None,
    auth_token=None,
):
    """
    Upload qc metrics to remote repository.

    :param str in_iqms: Path to the qc metric json file as a string
    :param str webapi_url: the protocol (either http or https)
    :param str email: email address to be included with the metric submission
    :param str auth_token: authentication token

    :return: either the response object if a response was successfully sent
             or it returns the string "No Response"
    :rtype: object


    """
    from copy import deepcopy
    from json import dumps, loads
    from pathlib import Path

    import requests

    if not endpoint or not auth_token:
        # If endpoint unknown, do not even report what happens to the token.
        errmsg = 'Unknown API endpoint' if not endpoint else 'Authentication failed.'
        return Bunch(status_code=1, text=errmsg)

    in_data = loads(Path(in_iqms).read_text())

    # Extract metadata and provenance
    meta = in_data.pop('bids_meta')

    # For compatibility with WebAPI. Should be rolled back to int
    if meta.get('run_id', None) is not None:
        meta['run_id'] = '%d' % meta.get('run_id')

    prov = in_data.pop('provenance')

    # At this point, data should contain only IQMs
    data = deepcopy(in_data)

    # Check modality
    modality = meta.get('modality', 'None')
    if modality not in ('T1w', 'bold', 'T2w'):
        errmsg = (
            'Submitting to MRIQCWebAPI: image modality should be "bold", "T1w", or "T2w", '
            '(found "%s")' % modality
        )
        return Bunch(status_code=1, text=errmsg)

    # Filter metadata values that aren't in whitelist
    data['bids_meta'] = {k: meta[k] for k in META_WHITELIST if k in meta}
    # Filter provenance values that aren't in whitelist
    data['provenance'] = {k: prov[k] for k in PROV_WHITELIST if k in prov}

    # Hash fields that may contain personal information
    data['bids_meta'] = _hashfields(data['bids_meta'])

    if email:
        data['provenance']['email'] = email

    headers = {'Authorization': auth_token, 'Content-Type': 'application/json'}

    start_message = messages.QC_UPLOAD_START.format(url=endpoint)
    config.loggers.interface.info(start_message)
    try:
        # if the modality is bold, call "bold" endpoint
        response = requests.post(
            f'{endpoint}/{modality}',
            headers=headers,
            data=dumps(data),
            timeout=15,
        )
    except requests.ConnectionError as err:
        errmsg = (
            'QC metrics failed to upload due to connection error shown below:\n%s' % err
        )
        return Bunch(status_code=1, text=errmsg)

    return response


def _hashfields(data):
    from hashlib import sha256

    for name in HASH_BIDS:
        if name in data:
            data[name] = sha256(data[name].encode()).hexdigest()

    return data
