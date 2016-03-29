#!/bin/bash

set -x
set -e

# Get test data
if [[ ! -d /scratch/data/ds003_downsampled ]]; then
    # Folder for downloads
    mkdir -p ${HOME}/downloads
    wget -P ${HOME}/downloads/ "https://googledrive.com/host/0B2JWN60ZLkgkMEw4bW5VUUpSdFU/ds003_downsampled.tar"
    mkdir -p /scratch/data/
    tar xf ${HOME}/downloads/ds003_downsampled.tar -C /scratch/data
fi

mkdir -p /scratch/data/ds003_sub-01
ln -fs /scratch/data/ds003_downsampled/sub-01 /scratch/data/ds003_sub-01/
echo "{plugin: MultiProc, plugin_args: {n_proc: 2}}" > /scratch/data/plugin.yml