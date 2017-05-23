#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# @Author: oesteban
# @Date:   2016-01-05 11:33:39
# @Email:  code@oscaresteban.es
# @Last modified by:   oesteban
# @Last Modified time: 2017-05-23 09:00:47
""" Helpers in report generation"""
from __future__ import print_function, division, absolute_import, unicode_literals


def iqms2html(indict, table_id):
    """Converts a dictionary into an HTML table"""
    columns = sorted(unfold_columns(indict))
    if not columns:
        return None

    depth = max([len(col) for col in columns])

    result_str = '<table id="%s">\n' % table_id
    td = '<td{1}>{0}</td>'.format
    for line in columns:
        result_str += '<tr>'
        ncols = len(line)
        for i, col in enumerate(line):
            colspan = 0
            colstring = ''
            if (depth - ncols) > 0 and i == ncols - 2:
                colspan = (depth - ncols) + 1
                colstring = ' colspan=%d' % colspan
            result_str += td(col, colstring)
        result_str += '</tr>\n'
    result_str += '</table>\n'
    return result_str

def unfold_columns(indict, prefix=None):
    """Converts an input dict with flattened keys to an array of columns"""
    if prefix is None:
        prefix = []
    keys = sorted(set(list(indict.keys())))

    data = []
    subdict = {}
    for key in keys:
        col = key.split('_', 1)
        if len(col) == 1:
            value = indict[col[0]]
            data.append(prefix + [col[0], value])
        else:
            if subdict.get(col[0]) is None:
                subdict[col[0]] = {}
            subdict[col[0]][col[1]] = indict[key]

    if subdict:
        for skey in sorted(list(subdict.keys())):
            sskeys = list(subdict[skey].keys())
            if len(sskeys) == 1:
                value = subdict[skey][sskeys[0]]
                newkey = '_'.join([skey] + sskeys)
                data.append(prefix + [newkey, value])
            else:
                data += unfold_columns(
                    subdict[skey], prefix=prefix + [skey])

    return data


def read_report_snippet(in_file):
    """Add a snippet into the report"""
    import os.path as op
    import re
    from io import open  #pylint: disable=W0622

    is_svg = (op.splitext(op.basename(in_file))[1] == '.svg')

    with open(in_file) as thisfile:
        if not is_svg:
            return thisfile.read()

        svg_tag_line = 0
        content = thisfile.read().split('\n')
        corrected = []
        for i, line in enumerate(content):
            if "<svg " in line:
                line = re.sub(' height="[0-9.]+[a-z]*"', '', line)
                line = re.sub(' width="[0-9.]+[a-z]*"', '', line)
                if svg_tag_line == 0:
                    svg_tag_line = i
            corrected.append(line)
        return '\n'.join(corrected[svg_tag_line:])

def upload_qc_metrics(in_iqms, email='', no_sub=False):
    """Upload qc metrics to remote repository.

    Arguments:
    in_iqms -- Path to the qc metric json file as a string

    Keyword arguments:
    email -- email address to be included with the metric submission, defaults to empty string
    no_sub -- Flag from settings indicating whether or not metrics should be submitted. If False, metrics will be submitted. If True, metrics will not be submitted. Defaults to False.

    Returns:
    either returns response object if a response was successfully sent
    or it returns the string "No Response"
    """
    from json import load, dumps
    import requests
    from mriqc import logging

    report_log = logging.getLogger('mriqc.report')
    report_log.setLevel(logging.INFO)

    if no_sub is True:
        report_log.info('QC metrics were not uploaded because --no_sub or --testing options were set.')
        r = "No Response"
    else:
        with open(in_iqms, 'r') as h:
            in_data = load(h)
        # metadata whitelist
        whitelist = ["ContrastBolusIngredient", "RepetitionTime", "TaskName", "Manufacturer",
                     "ManufacturersModelName", "MagneticFieldStrength", "DeviceSerialNumber",
                     "SoftwareVersions", "HardcopyDeviceSoftwareVersion", "ReceiveCoilName",
                     "GradientSetType", "MRTransmitCoilSequence", "MatrixCoilMode",
                     "CoilCombinationMethod", "PulseSequenceType", "PulseSequenceDetails",
                     "NumberShots", "ParallelReductionFactorInPlane", "ParallelAcquisitionTechnique",
                     "PartialFourier", "PartialFourierDirection", "PhaseEncodingDirection",
                     "EffectiveEchoSpacing", "TotalReadoutTime",
                     "EchoTime", "InversionTime", "SliceTiming", "SliceEncodingDirection",
                     "NumberOfVolumesDiscardedByScanner", "NumberOfVolumesDiscardedByUser",
                     "DelayTime", "FlipAngle", "MultibandAccelerationFactor", "Instructions",
                     "TaskDescription", "CogAtlasID", "CogPOID", "InstitutionName",
                     "InstitutionAddress", "ConversionSoftware", "ConversionSoftwareVersion",
                     "md5sum", "modality", "mriqc_pred", "software", "subject_id", "version"]
        # flatten data
        data = {k: v for k, v in list(in_data.items()) if k != 'metadata'}
        # Filter Metadata values that aren't in whitelist
        try:
            data.update({k: v for k, v in list(in_data['metadata'].items()) if k in whitelist})
        except KeyError:
            pass
        # Preemptively adding code to handle settings
        try:
            data.update({k: v for k, v in list(in_data['settings'].items()) if k in whitelist})
        except KeyError:
            pass

        if email != '':
            data['email'] = email
        secret_key = 'ZUsBaabr6PEbav5DKAHIODEnwpwC58oQTJF7KWvDBPUmBIVFFtwOd7lQBdz9r9ulJTR1BtxBDqDuY0owxK6LbLB1u1b64ZkIMd46'
        headers = {'token': secret_key, "Content-Type": "application/json"}
        try:
            r = requests.put("http://34.201.213.252:5000/measurements/upload",
                             headers=headers, data=dumps(data))
            if r.status_code == 201:
                report_log.info('QC metrics successfully uploaded.')
            else:
                report_log.warn('QC metrics failed to upload. Status %d: %s' % (r.status_code, r.text))
        except requests.ConnectionError as e:
            report_log.warn('QC metrics failed to upload due to connection error shown below:\n%s' % e)
            r = "No Response"
    return r


# def check_reports(dataset, settings, save_failed=True):
#     """Check if reports have been created"""
#     import os.path as op
#     import pandas as pd
#     from mriqc.utils.misc import BIDS_COMP, BIDS_EXPR
#     supported_components = list(BIDS_COMP.keys())
#     expr = re.compile(BIDS_EXPR)

#     reports_missed = False
#     missing = {}
#     for mod, files in list(dataset.items()):
#         missing[mod] = []
#         qctype = 'anatomical' if mod == 't1w' else 'functional'

#         for fname in files:
#             m = expr.search(op.basename(fname)).groupdict()
#             components = [m.get(key) for key in supported_components if m.get(key)]
#             components.insert(0, qctype)

#             report_fname = op.join(
#                 settings['report_dir'], '_'.join(components) + '_report.html')

#             if not op.isfile(report_fname):
#                 missing[mod].append(
#                     {key: m.get(key) for key in supported_components if m.get(key)})

#         mod_missing = missing[mod]
#         if mod_missing:
#             reports_missed = True

#         if mod_missing and save_failed:
#             out_file = op.join(settings['output_dir'], 'failed_%s.csv' % qctype)
#             miss_cols = list(set(supported_components) & set(list(mod_missing[0].keys())))
#             dframe = pd.DataFrame.from_dict(mod_missing).sort_values(
#                 by=miss_cols)
#             dframe[miss_cols].to_csv(out_file, index=False)

#     return reports_missed
