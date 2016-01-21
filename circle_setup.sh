#!/bin/bash

set -x
set -e

# fsldir=${HOME}/.local/fsl
# afnidir=${HOME}/.local/afni

# if [[ ! -s ${fsldir}/etc/fslconf/fsl.sh ]]; then 
#     # Download md5 checksums
#     echo "69fb1622043e11ad678900bfb5f93c14  fsl.tar.gz" > ${HOME}/.local/downloads/fsl.tar.gz.md5

#     mkdir -p ${HOME}/.local/downloads
#     cd ${HOME}/.local/downloads
#     fslchk=$(md5sum -c fsl.tar.gz.md5 | awk '{print $2}')
#     cd

#     # Get fsl if md5 is not ok
#     if [ $fslchk != "OK" ]; then 
#         wget -O ${HOME}/.local/downloads/fsl.tar.gz -c "http://fsl.fmrib.ox.ac.uk/fsldownloads/fsl-5.0.9-centos6_64.tar.gz"
#         rm -rf ${fsldir}
#     fi

#     tar zxvf ${HOME}/.local/downloads/fsl.tar.gz -C ${HOME}/.local/
# fi

# source ${fsldir}/etc/fslconf/fsl.sh

# if [[ ! -d ${afnidir} ]]; then
#     echo "532323af582845c38517da1e93f8fc99  afni.tgz" > ${HOME}/.local/downloads/afni.tgz.md5
#     mkdir -p ${HOME}/.local/downloads
#     cd ${HOME}/.local/downloads
#     afnichk=$(md5sum -c afni.tgz.md5 | awk '{print $2}')
#     cd

#     # Get afni if md5 is not ok
#     if [ $afnichk != "OK" ]; then
#         wget -O ${HOME}/.local/downloads/afni.tgz.md5 -c "http://afni.nimh.nih.gov/pub/dist/tgz/linux_openmp_64.tgz"
#         rm -rf ${afnidir}
#     fi

#     tar zxvf ${HOME}/.local/downloads/afni.tgz -C ${HOME}
#     mv ${HOME}/linux_openmp_64 ${afnidir}
# fi

# Get test data
if [[ ! -d ${HOME}/examples/ds003_downsampled ]]; then
    # Folder for downloads
    mkdir -p ${HOME}/downloads
    wget -P ${HOME}/downloads/ "https://googledrive.com/host/0B2JWN60ZLkgkMEw4bW5VUUpSdFU/ds003_downsampled.tar"
    mkdir -p ${HOME}/examples
    tar xvf ${HOME}/downloads/ds003_downsampled.tar -C ${HOME}/examples
fi

mkdir -p ~/examples/ds003_sub-01
ln -fs ~/examples/ds003_downsampled/sub-01 ~/examples/ds003_sub-01/
