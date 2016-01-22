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

mkdir -p ~/examples/ds003_sub-01
ln -fs ~/examples/ds003_downsampled/sub-01 ~/examples/ds003_sub-01/
echo "{plugin: MultiProc, plugin_args: {n_proc: 4}}" > ~/examples/plugin.yml

mkdir -p ~/workdir/
mkdir -p ~/outdir/
