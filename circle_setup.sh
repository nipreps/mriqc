#!/bin/bash

set -x
set -e

# Get test data
if [[ ! -d ${HOME}/examples/ds003_downsampled ]]; then
    # Folder for downloads
    mkdir -p ${HOME}/downloads
    wget -P ${HOME}/downloads/ "https://googledrive.com/host/0B2JWN60ZLkgkMEw4bW5VUUpSdFU/ds003_downsampled.tar"
    mkdir -p ${HOME}/examples
    tar xf ${HOME}/downloads/ds003_downsampled.tar -C ${HOME}/examples
fi

if [[ ! -d ${HOME}/local/ants ]]; then
	mkdir -p ${HOME}/downloads
	wget -c -O ${HOME}/downloads/ants.tar.bz2 "https://2a353b13e8d2d9ac21ce543b7064482f771ce658.googledrive.com/host/0BxI12kyv2olZVFhUcGVpYWF3R3c/ANTs-Linux_Ubuntu14.04.tar.bz2"
	mkdir -p ${HOME}/local/ants
	tar xjf ${HOME}/downloads/ants.tar.bz2 -C ${HOME}/local/
	mv ${HOME}/local/ANTs.2.1.0.Debian-Ubuntu_X64 ${HOME}/local/ants
fi

mkdir -p ~/examples/ds003_sub-01
ln -fs ~/examples/ds003_downsampled/sub-01 ~/examples/ds003_sub-01/
echo "{plugin: MultiProc, plugin_args: {n_proc: 4}}" > ~/examples/plugin.yml

mkdir -p ~/workdir/
mkdir -p ~/outdir/
