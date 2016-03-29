

FROM ubuntu:vivid

MAINTAINER Oscar Esteban <code@oscaresteban.es>

RUN ln -snf /bin/bash /bin/sh

# Update packages and install the minimal set of tools
RUN apt-get update && \
    apt-get install -y curl git xvfb bzip2 apt-utils

# replace .bashrc
# ADD files/bashrc /root/.bashrc

# Install ANTs
RUN mkdir -p /opt/ants && \
    curl -sSL "https://2a353b13e8d2d9ac21ce543b7064482f771ce658.googledrive.com/host/0BxI12kyv2olZVFhUcGVpYWF3R3c/ANTs-Linux_Ubuntu14.04.tar.bz2" \
    | tar -xjC /opt/ants --strip-components 1
ENV PATH /opt/ants:$PATH

# Enable neurodebian
RUN curl -sSL http://neuro.debian.net/lists/vivid.de-m.full | tee /etc/apt/sources.list.d/neurodebian.sources.list && \
    curl -sSL http://neuro.debian.net/lists/vivid.us-tn.full >> /etc/apt/sources.list.d/neurodebian.sources.list && \
    apt-key adv --recv-keys --keyserver hkp://pgp.mit.edu:80 0xA5D32F012649A5A9 && \
    apt-get update && \
    apt-get install -y fsl-core afni

# Clear apt cache to reduce image size
RUN rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Install wrapper
ADD files/run_mriqc /usr/bin/run_mriqc
RUN chmod +x /usr/bin/run_mriqc
# RUN groupadd -r ubuntu && useradd -m -s /bin/bash -g ubuntu ubuntu && \
#     mkdir -p /home/ubuntu/data && \
#     chown ubuntu.ubuntu /home/ubuntu/data
# 
# USER ubuntu
# WORKDIR /home/ubuntu

WORKDIR /root

# Install miniconda
RUN curl -sSLO https://repo.continuum.io/miniconda/Miniconda-latest-Linux-x86_64.sh && \
    /bin/bash Miniconda-latest-Linux-x86_64.sh -b && \
    rm Miniconda-latest-Linux-x86_64.sh

ENV PATH /root/miniconda2/bin:$PATH

# http://bugs.python.org/issue19846
# > At the moment, setting "LANG=C" on a Linux system *fundamentally breaks Python 3*, and that's not OK.
ENV LANG C.UTF-8

# Create conda environment
RUN conda create -y -n mriqcenv numpy \
                                scipy \
                                pandas \
                                scikit-image \
                                ipython \
                                matplotlib \
                                networkx \
                                lxml

# Install pip
RUN source activate mriqcenv && \
    pip install --upgrade pip && \
    pip install xvfbwrapper && \
    pip install lockfile && \
    pip install --upgrade numpy && \
    pip install --upgrade pandas

# Install nipype & mriqc
RUN mkdir -p src && \
    cd src && \
    git clone https://github.com/nipy/nipype.git && \
    git clone https://github.com/poldracklab/mriqc.git && \
    source activate mriqcenv && \
    cd nipype && \
    pip install -e . && \
    cd ../mriqc && \
    git pull && \
    pip install -e .

ADD files/bashrc /root/.bashrc

ENTRYPOINT ["/usr/bin/run_mriqc"]
CMD ["--help"]
