#!/bin/bash

set -x
set -e

# Get test data
if [[ ! -d ${HOME}/scratch/data/ds003_downsampled ]]; then
    # Folder for downloads
    mkdir -p ${HOME}/downloads
    wget -c -P ${HOME}/downloads/ "https://googledrive.com/host/0B2JWN60ZLkgkMEw4bW5VUUpSdFU/ds003_downsampled.tar"
    mkdir -p ${HOME}/scratch/data/
    tar xf ${HOME}/downloads/ds003_downsampled.tar -C ${HOME}/scratch/data
fi

mkdir -p ${HOME}/scratch/data/ds003_sub-01
ln -fs ${HOME}/scratch/data/ds003_downsampled/sub-01 ${HOME}/scratch/data/ds003_sub-01/
echo "{plugin: MultiProc, plugin_args: {n_proc: 4}}" > ${HOME}/scratch/plugin.yml