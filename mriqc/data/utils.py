#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# @Author: oesteban
# @Date:   2016-01-05 11:29:40
# @Email:  code@oscaresteban.es
# @Last modified by:   oesteban
# @Last Modified time: 2016-04-06 12:07:51
"""
Utilities for data grabbers (from nilearn)
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import os
import os.path as op
import sys
import shutil
import time
import base64
import hashlib
import subprocess as sp
from mriqc.data.compat import _urllib, md5_hash


def _fetch_file(url, dataset_dir, filetype=None, resume=True, overwrite=False,
                md5sum=None, username=None, password=None, handlers=None,
                verbose=1, temp_downloads=None):
    """Load requested file, downloading it if needed or requested.

    :param str url: contains the url of the file to be downloaded.
    :param str dataset_dir: path of the data directory. Used for data
        storage in the specified location.
    :param bool resume: if true, try to resume partially downloaded files
    :param overwrite: if bool true and file already exists, delete it.
    :param str md5sum: MD5 sum of the file. Checked if download of the file
        is required
    :param str username: username used for basic HTTP authentication
    :param str password: password used for basic HTTP authentication
    :param list(BaseHandler) handlers: urllib handlers passed to
        urllib.request.build_opener. Used by advanced users to customize
        request handling.
    :param int verbose: verbosity level (0 means no message).
    :returns: absolute path of downloaded file
    :rtype: str

    ..note::

      If, for any reason, the download procedure fails, all downloaded files are
      removed.


    """
    if handlers is None:
        handlers = []

    if not overwrite and os.listdir(dataset_dir):
        return True

    data_dir, _ = op.split(dataset_dir)

    if temp_downloads is None:
        temp_downloads = op.join(op.expanduser('~/.cache/mriqc'), 'downloads')

    # Determine data path
    if not op.exists(temp_downloads):
        os.makedirs(temp_downloads)

    # Determine filename using URL
    parse = _urllib.parse.urlparse(url)
    file_name = op.basename(parse.path)
    if file_name == '':
        file_name = md5_hash(parse.path)

        if filetype is not None:
            file_name += filetype

    temp_full_name = op.join(temp_downloads, file_name)
    temp_part_name = temp_full_name + ".part"

    if overwrite:
        shutil.rmtree(dataset_dir, ignore_errors=True)

    if op.exists(temp_full_name):
        if overwrite:
            os.remove(temp_full_name)

    t_0 = time.time()
    local_file = None
    initial_size = 0

    try:
        # Download data
        url_opener = _urllib.request.build_opener(*handlers)
        request = _urllib.request.Request(url)
        request.add_header('Connection', 'Keep-Alive')
        if username is not None and password is not None:
            if not url.startswith('https'):
                raise ValueError(
                    'Authentication was requested on a non  secured URL (%s).'
                    'Request has been blocked for security reasons.' % url)
            # Note: HTTPBasicAuthHandler is not fitted here because it relies
            # on the fact that the server will return a 401 error with proper
            # www-authentication header, which is not the case of most
            # servers.
            encoded_auth = base64.b64encode(
                (username + ':' + password).encode())
            request.add_header(b'Authorization', b'Basic ' + encoded_auth)
        if verbose > 0:
            displayed_url = url.split('?')[0] if verbose == 1 else url
            print('Downloading data from %s ...' % displayed_url)
        if resume and op.exists(temp_part_name):
            # Download has been interrupted, we try to resume it.
            local_file_size = op.getsize(temp_part_name)
            # If the file exists, then only download the remainder
            request.add_header("Range", "bytes=%s-" % (local_file_size))
            try:
                data = url_opener.open(request)
                content_range = data.info().get('Content-Range')
                if (content_range is None or not content_range.startswith(
                        'bytes %s-' % local_file_size)):
                    raise IOError('Server does not support resuming')
            except Exception:
                # A wide number of errors can be raised here. HTTPError,
                # URLError... I prefer to catch them all and rerun without
                # resuming.
                if verbose > 0:
                    print('Resuming failed, try to download the whole file.')
                return _fetch_file(
                    url, dataset_dir, resume=False, overwrite=overwrite,
                    md5sum=md5sum, username=username, password=password,
                    handlers=handlers, verbose=verbose)
            local_file = open(temp_part_name, "ab")
            initial_size = local_file_size
        else:
            data = url_opener.open(request)
            local_file = open(temp_part_name, "wb")
        _chunk_read_(data, local_file, report_hook=(verbose > 0),
                     initial_size=initial_size, verbose=verbose)
        # temp file must be closed prior to the move
        if not local_file.closed:
            local_file.close()
        shutil.move(temp_part_name, temp_full_name)
        delta_t = time.time() - t_0
        if verbose > 0:
            # Complete the reporting hook
            sys.stderr.write(' ...done. ({0:.0f} seconds, {1:.0f} min)\n'
                             .format(delta_t, delta_t // 60))
    except (_urllib.error.HTTPError, _urllib.error.URLError) as exc:
        if 'Error while fetching' not in str(exc):
            # For some odd reason, the error message gets doubled up
            #   (possibly from the re-raise), so only add extra info
            #   if it's not already there.
            exc.reason = ("%s| Error while fetching file %s; "
                          "dataset fetching aborted." % (
                              str(exc.reason), file_name))
        raise
    finally:
        if local_file is not None:
            if not local_file.closed:
                local_file.close()

    if md5sum is not None:
        if _md5_sum_file(temp_full_name) != md5sum:
            raise ValueError("File %s checksum verification has failed."
                             " Dataset fetching aborted." % local_file)

    if filetype is None:
        fname, filetype = op.splitext(op.basename(temp_full_name))
        if filetype == '.gz':
            fname, ext = op.splitext(fname)
            filetype = ext + filetype

    if filetype.startswith('.'):
        filetype = filetype[1:]

    if filetype == 'tar':
        sp.check_call(['tar', 'xf', temp_full_name], cwd=data_dir)
        os.remove(temp_full_name)
        return True

    return True

def _get_dataset_dir(dataset_name, data_dir=None, default_paths=None,
                     verbose=1):
    """ Create if necessary and returns data directory of given dataset.

    :param str dataset_name: The unique name of the dataset.
    :param str data_dir: Path of the data directory. Used to force data storage in
      a specified location.
    :param list(str) default_paths: Default system paths in which the dataset
      may already have been installed by a third party software. They will be
      checked first.
    :param int verbose: verbosity level (0 means no message).

    :returns: the path of the given dataset directory.
    :rtype: str

    .. note::

      This function retrieves the datasets directory (or data directory) using
      the following priority :

        1. defaults system paths
        2. the keyword argument data_dir
        3. the global environment variable MRIQC_SHARED_DATA
        4. the user environment variable MRIQC_DATA
        5. ~/.cache/mriqc in the user home folder


    """
    # We build an array of successive paths by priority
    # The boolean indicates if it is a pre_dir: in that case, we won't add the
    # dataset name to the path.
    paths = []

    # Check data_dir which force storage in a specific location
    if data_dir is not None:
        paths.extend([(d, False) for d in data_dir.split(os.pathsep)])

    # Search possible system paths
    if default_paths is not None:
        for default_path in default_paths:
            paths.extend([(d, True) for d in default_path.split(os.pathsep)])

    # If data_dir has not been specified, then we crawl default locations
    if data_dir is None:
        global_data = os.getenv('MRIQC_SHARED_DATA')
        if global_data is not None:
            paths.extend([(d, False) for d in global_data.split(os.pathsep)])

        local_data = os.getenv('MRIQC_DATA')
        if local_data is not None:
            paths.extend([(d, False) for d in local_data.split(os.pathsep)])

        paths.append((op.expanduser('~/.cache/mriqc'), False))

    if verbose > 2:
        print('Dataset search paths: %s' % paths)

    # Check if the dataset exists somewhere
    for path, is_pre_dir in paths:
        if not is_pre_dir:
            path = op.join(path, dataset_name)
        if op.islink(path):
            # Resolve path
            path = readlinkabs(path)
        if op.exists(path) and op.isdir(path):
            if verbose > 1:
                print('\nDataset found in %s\n' % path)
            return path

    # If not, create a folder in the first writeable directory
    errors = []
    for (path, is_pre_dir) in paths:
        if not is_pre_dir:
            path = op.join(path, dataset_name)
        if not op.exists(path):
            try:
                os.makedirs(path)
                if verbose > 0:
                    print('\nDataset created in %s\n' % path)
                return path
            except Exception as exc:
                short_error_message = getattr(exc, 'strerror', str(exc))
                errors.append('\n -{0} ({1})'.format(
                    path, short_error_message))

    raise OSError('mriqc tried to store the dataset in the following '
                  'directories, but:' + ''.join(errors))


def readlinkabs(link):
    """
    Return an absolute path for the destination
    of a symlink
    """
    path = os.readlink(link)
    if op.isabs(path):
        return path
    return op.join(op.dirname(link), path)


def _md5_sum_file(path):
    """ Calculates the MD5 sum of a file.
    """
    with open(path, 'rb') as fhandle:
        md5sum = hashlib.md5()
        while True:
            data = fhandle.read(8192)
            if not data:
                break
            md5sum.update(data)
    return md5sum.hexdigest()


def _chunk_read_(response, local_file, chunk_size=8192, report_hook=None,
                 initial_size=0, total_size=None, verbose=1):
    """Download a file chunk by chunk and show advancement

    :param _urllib.response.addinfourl response: response to the download
        request in order to get file size
    :param str local_file: hard disk file where data should be written
    :param int chunk_size: size of downloaded chunks. Default: 8192
    :param bool report_hook: whether or not to show downloading advancement
    :param int initial_size: if resuming, indicate the initial size of the file
    :param int total_size: Expected final size of download (None means it
        is unknown).
    :param int verbose: verbosity level (0 means no message).
    :returns: the downloaded file path.
    :rtype: string

    """
    try:
        if total_size is None:
            total_size = response.info().get('Content-Length').strip()
        total_size = int(total_size) + initial_size
    except Exception as exc:
        if verbose > 2:
            print("Warning: total size could not be determined.")
            if verbose > 3:
                print("Full stack trace: %s" % exc)
        total_size = None
    bytes_so_far = initial_size

    t_0 = time_last_display = time.time()
    while True:
        chunk = response.read(chunk_size)
        bytes_so_far += len(chunk)
        time_last_read = time.time()
        if (report_hook and
                # Refresh report every half second or when download is
                # finished.
                (time_last_read > time_last_display + 0.5 or not chunk)):
            _chunk_report_(bytes_so_far,
                           total_size, initial_size, t_0)
            time_last_display = time_last_read
        if chunk:
            local_file.write(chunk)
        else:
            break

    return

def _chunk_report_(bytes_so_far, total_size, initial_size, t_0):
    """Show downloading percentage.

    :param int bytes_so_far: number of downloaded bytes
    :param int total_size: total size of the file (may be 0/None, depending
        on download method).
    :param int t_0: the time in seconds (as returned by time.time()) at which
        the download was resumed / started.
    :param int initial_size: if resuming, indicate the initial size of the
        file. If not resuming, set to zero.
    """

    if not total_size:
        sys.stderr.write("\rDownloaded %d of ? bytes." % (bytes_so_far))

    else:
        # Estimate remaining download time
        total_percent = float(bytes_so_far) / total_size

        current_download_size = bytes_so_far - initial_size
        bytes_remaining = total_size - bytes_so_far
        delta_t = time.time() - t_0
        download_rate = current_download_size / max(1e-8, float(delta_t))
        # Minimum rate of 0.01 bytes/s, to avoid dividing by zero.
        time_remaining = bytes_remaining / max(0.01, download_rate)

        # Trailing whitespace is to erase extra char when message length
        # varies
        sys.stderr.write(
            "\rDownloaded %d of %d bytes (%.1f%%, %s remaining)"
            % (bytes_so_far, total_size, total_percent * 100,
               _format_time(time_remaining)))


def _format_time(t_secs):
    if t_secs > 60:
        return "%4.1fmin" % (t_secs / 60.)
    else:
        return " %5.1fs" % (t_secs)
