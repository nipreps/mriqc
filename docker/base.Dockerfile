# Copyright (c) 2016, The developers of the Stanford CRN
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of crn_base nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


#
# Based on https://github.com/poldracklab/fmriprep/blob/9c92a3de9112f8ef1655b876de060a2ad336ffb0/Dockerfile
#
FROM ubuntu:xenial-20161213
MAINTAINER The CRN developers

ARG DEBIAN_FRONTEND=noninteractive

# Pre-cache neurodebian key
COPY docker/files/neurodebian.gpg /root/.neurodebian.gpg

# Prepare environment
RUN apt-key add /root/.neurodebian.gpg && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
                    curl \
                    xvfb \
                    bzip2 \
                    ca-certificates \
                    unzip \
                    apt-utils \
                    fusefat \
                    make \
                    gcc \
                    git=1:2.7.4-0ubuntu1 \
                    graphviz=2.38.0-12ubuntu2 \
                    nodejs && \
    curl -sSL http://neuro.debian.net/lists/xenial.us-ca.full >> /etc/apt/sources.list.d/neurodebian.sources.list && \
    apt-key adv --refresh-keys --keyserver hkp://ha.pool.sks-keyservers.net 0xA5D32F012649A5A9 || true; \
    apt-get update

# Installing Neurodebian packages (FSL, AFNI, git)
RUN apt-get install -y --no-install-recommends \
                    fsl-core=5.0.9-1~nd+1+nd16.04+1 \
                    fsl-mni152-templates=5.0.7-2 \
                    afni=16.2.07~dfsg.1-2~nd16.04+1 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

ENV FSLDIR=/usr/share/fsl/5.0 \
    FSLOUTPUTTYPE=NIFTI_GZ \
    FSLMULTIFILEQUIT=TRUE \
    POSSUMDIR=/usr/share/fsl/5.0 \
    LD_LIBRARY_PATH=/usr/lib/fsl/5.0:$LD_LIBRARY_PATH \
    FSLTCLSH=/usr/bin/tclsh \
    FSLWISH=/usr/bin/wish \
    AFNI_MODELPATH=/usr/lib/afni/models \
    AFNI_IMSAVE_WARNINGS=NO \
    AFNI_TTATLAS_DATASET=/usr/share/afni/atlases \
    AFNI_PLUGINPATH=/usr/lib/afni/plugins \
    PATH=/usr/lib/fsl/5.0:/usr/lib/afni/bin:$PATH

# Installing and setting up ANTs
RUN mkdir -p /opt/ants && \
    curl -sSL "https://github.com/stnava/ANTs/releases/download/v2.1.0/Linux_Ubuntu14.04.tar.bz2" \
    | tar -xjC /opt/ants --strip-components 1

ENV PATH=/opt/ants:$PATH

# Installing and setting up c3d
RUN mkdir -p /opt/c3d && \
    curl -sSL "https://files.osf.io/v1/resources/fvuh8/providers/osfstorage/57f341d6594d9001f591bac2" \
    | tar -xzC /opt/c3d --strip-components 1

ENV PATH=/opt/c3d:$PATH

# Install BIDS-validator
RUN curl -sL https://deb.nodesource.com/setup_4.x | bash - && \
    apt-get install -y nodejs && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* && \
    npm install -g bids-validator

