#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
from __future__ import print_function, division, absolute_import, unicode_literals

from niworkflows.nipype import logging
from niworkflows.nipype.interfaces.base import (
    Bunch, traits, isdefined, TraitedSpec, BaseInterfaceInputSpec, File, Str,
    SimpleInterface
)
from urllib.parse import urlparse

IFLOGGER = logging.getLogger('interface')

SECRET_KEY = '<secret_token>'

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

PROV_WHITELIST = [
    'version',
    'md5sum',
    'software',
    'settings'
]

HASH_BIDS = ['subject_id', 'session_id']


class UploadIQMsInputSpec(BaseInterfaceInputSpec):
    in_iqms = File(exists=True, mandatory=True, desc='the input IQMs-JSON file')
    url = Str(mandatory=True, desc='URL (protocol and name) listening')
    port = traits.Int(desc='MRIQCWebAPI service port')
    path = Str(desc='MRIQCWebAPI endpoint root path')
    email = Str(desc='set sender email')
    strict = traits.Bool(False, usedefault=True,
                         desc='crash if upload was not succesfull')


class UploadIQMs(SimpleInterface):
    """
    Upload features to MRIQCWebAPI
    """

    input_spec = UploadIQMsInputSpec
    output_spec = TraitedSpec
    always_run = True

    def _run_interface(self, runtime):
        email = None
        if isdefined(self.inputs.email):
            email = self.inputs.email

        rawurl = self.inputs.url
        if '://' not in rawurl:
            rawurl = 'http://'
        url = urlparse(rawurl)

        if not url.scheme.startswith('http'):
            raise RuntimeError(
                'Tried an unknown protocol "%s"' % url.scheme)

        port = url.port
        if isdefined(self.inputs.port):
            port = self.inputs.port

        path = url.path
        if isdefined(self.inputs.path):
            path = self.inputs.path

        response = upload_qc_metrics(
            self.inputs.in_iqms, url.netloc, path=path,
            scheme=url.scheme, port=port, email=email)

        if response.status_code == 201:
            IFLOGGER.info('QC metrics successfully uploaded.')
            return runtime

        errmsg = 'QC metrics failed to upload. Status %d: %s' % (
            response.status_code, response.text)
        IFLOGGER.warn(errmsg)
        if self.inputs.strict:
            raise RuntimeError(response.text)

        return runtime


def upload_qc_metrics(in_iqms, loc, path='', scheme='http',
                      port=None, email=None):
    """
    Upload qc metrics to remote repository.

    :param str in_iqms: Path to the qc metric json file as a string
    :param str scheme: the protocol (either http or https)
    :param str email: email address to be included with the metric submission
    :param bool upload_strict: the client should fail if it's strict mode

    :return: either the response object if a response was successfully sent
             or it returns the string "No Response"
    :rtype: object


    """
    from json import load, dumps
    import requests
    from io import open
    from copy import deepcopy

    if port is None:
        port = 443 if scheme == 'https' else 80

    with open(in_iqms, 'r') as input_json:
        in_data = load(input_json)

    # Extract metadata and provenance
    meta = in_data.pop('bids_meta')
    prov = in_data.pop('provenance')

    # At this point, data should contain only IQMs
    data = deepcopy(in_data)

    # Check modality
    modality = meta.get('modality', 'None')
    if modality not in ('T1w', 'bold', 'T2w'):
        errmsg = ('Submitting to MRIQCWebAPI: image modality should be "bold", "T1w", or "T2w", '
                  '(found "%s")' % modality)
        return Bunch(status_code=1, text=errmsg)

    # Filter metadata values that aren't in whitelist
    data['bids_meta'] = {k: meta[k] for k in META_WHITELIST if k in meta}
    # Filter provenance values that aren't in whitelist
    data['provenance'] = {k: prov[k] for k in PROV_WHITELIST if k in prov}

    # Hash fields that may contain personal information
    data['bids_meta'] = _hashfields(data['bids_meta'])

    if email:
        data['provenance']['email'] = email

    if path and not path.endswith('/'):
        path += '/'
        if path.startswith('/'):
            path = path[1:]

    headers = {'Authorization': SECRET_KEY, "Content-Type": "application/json"}

    webapi_url = '{}://{}:{}/{}{}'.format(scheme, loc, port, path, modality)
    IFLOGGER.info('MRIQC Web API: submitting to <%s>', webapi_url)
    try:
        # if the modality is bold, call "bold" endpoint
        response = requests.post(webapi_url, headers=headers, data=dumps(data))
    except requests.ConnectionError as err:
        errmsg = 'QC metrics failed to upload due to connection error shown below:\n%s' % err
        return Bunch(status_code=1, text=errmsg)

    return response


def _hashfields(data):
    from hashlib import sha256

    for name in HASH_BIDS:
        if name in data:
            data[name] = sha256(data[name].encode()).hexdigest()

    return data
