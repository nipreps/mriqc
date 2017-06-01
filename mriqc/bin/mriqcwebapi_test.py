#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: oesteban
# @Date:   2015-11-19 16:44:27
from __future__ import print_function, division, absolute_import, unicode_literals

def get_parser():
    """Build parser object"""
    from argparse import ArgumentParser
    from argparse import RawTextHelpFormatter

    parser = ArgumentParser(description='MRIQCWebAPI: Check entries',
                            formatter_class=RawTextHelpFormatter)
    parser.add_argument('modality', action='store', choices=['T1w', 'bold'],
    	                help='number of expected items in the database')
    parser.add_argument('expected', action='store', type=int,
    	                help='number of expected items in the database')
    parser.add_argument(
        '--webapi-addr', action='store', default='34.201.213.252', type=str,
        help='IP address where the MRIQC WebAPI is listening')
    parser.add_argument(
        '--webapi-port', action='store', default=80, type=int,
        help='port where the MRIQC WebAPI is listening')
    return parser


def main():
    """Entry point"""
    from requests import get
    from mriqc import MRIQC_LOG

    # Run parser
    opts = get_parser().parse_args()

    endpoint = 'http://{}:{}/{}'.format(opts.webapi_addr,
                                        opts.webapi_port,
                                        opts.modality)
    MRIQC_LOG.info('Sending GET: %s', endpoint)
    resp = get(endpoint).json()
    MRIQC_LOG.info('There are %d records in database', resp['_meta']['total'])
    assert opts.expected == resp['_meta']['total']


if __name__ == '__main__':
    main()
