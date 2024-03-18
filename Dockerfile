# MRIQC Docker Container Image distribution
#
# MIT License
#
# Copyright (c) 2021 The NiPreps Developers
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Ubuntu 22.04 LTS - Jammy
ARG BASE_IMAGE=ubuntu:jammy-20240125

#
# Build wheel
#
FROM python:slim AS src
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive \
    apt-get install -y --no-install-recommends git
RUN python -m pip install -U pip build
COPY . /src
RUN python -m build /src

# Utilities for downloading packages
FROM ${BASE_IMAGE} as downloader
# Bump the date to current to refresh curl/certificates/etc
RUN echo "2024.03.18"
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive \
    apt-get install -y --no-install-recommends \
                    binutils \
                    bzip2 \
                    ca-certificates \
                    curl \
                    unzip && \
    apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*


# AFNI
FROM downloader as afni
# Bump the date to current to update AFNI
RUN echo "2024.03.18"
RUN mkdir -p /opt/afni-latest \
    && curl -fsSL --retry 5 https://afni.nimh.nih.gov/pub/dist/tgz/linux_openmp_64.tgz \
    | tar -xz -C /opt/afni-latest --strip-components 1 \
    --exclude "linux_openmp_64/*.gz" \
    --exclude "linux_openmp_64/funstuff" \
    --exclude "linux_openmp_64/shiny" \
    --exclude "linux_openmp_64/afnipy" \
    --exclude "linux_openmp_64/lib/RetroTS" \
    --exclude "linux_openmp_64/lib_RetroTS" \
    --exclude "linux_openmp_64/meica.libs" \
    # Keep only what we use
    && find /opt/afni-latest -type f -not \( \
        -name "afni" -or \
        -name "3dAutomask" -or \
        -name "3dcalc" -or \
        -name "3dSkullStrip" -or \
        -name "3dToutcount" -or \
        -name "3dTqual" -or \
        -name "3dTshift" -or \
        -name "3dTstat" -or \
        -name "3dUnifize" -or \
        -name "3dvolreg" \) -delete

# Use Ubuntu 20.04 LTS
FROM nipreps/miniconda:py39_2403.0

ARG DEBIAN_FRONTEND=noninteractive
ENV LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:${CONDA_PATH}/lib"
ENV CONDA_PATH="/opt/conda"

# Install AFNI
ENV AFNI_DIR="/opt/afni"
COPY --from=afni /opt/afni-latest ${AFNI_DIR}
ENV PATH="${AFNI_DIR}:$PATH" \
    AFNI_IMSAVE_WARNINGS="NO" \
    AFNI_MODELPATH="${AFNI_DIR}/models" \
    AFNI_TTATLAS_DATASET="${AFNI_DIR}/atlases" \
    AFNI_PLUGINPATH="${AFNI_DIR}/plugins"

RUN apt-get update \
	 && DEBIAN_FRONTEND=noninteractive \
	    apt-get install -y -q --no-install-recommends     \
			    libcurl4-openssl-dev              \
			    libgdal-dev                       \
			    libgfortran-12-dev                \
			    libgfortran5                      \
			    libglw1-mesa                      \
			    libgomp1                          \
			    libjpeg62                         \
			    libnode-dev                       \
			    libssl-dev                        \
			    libudunits2-dev                   \
			    libxm4                            \
			    libxml2-dev                       \
			    netbase                           \
			    netpbm                            \
			    tcsh                              \
			    xfonts-base                       \
	 && apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* \
	 && ldconfig

# Install AFNI's dependencies
RUN micromamba install -n base -c conda-forge -c anaconda \
                       gsl                                \
                       xorg-libxp                         \
	    && micromamba install -n base -c conda-forge "ants=2.5" \
            && sync \
	    && micromamba clean -afy; sync \
	    && ln -s ${CONDA_PATH}/lib/libgsl.so.25 /usr/lib/x86_64-linux-gnu/libgsl.so.19 \
	    && ln -s ${CONDA_PATH}/lib/libgsl.so.25 /usr/lib/x86_64-linux-gnu/libgsl.so.0 \
	    && ldconfig


# Unless otherwise specified each process should only use one thread - nipype
# will handle parallelization
ENV MKL_NUM_THREADS=1 \
    OMP_NUM_THREADS=1

COPY --from=freesurfer/synthstrip@sha256:f19578e5f033f2c707fa66efc8b3e11440569facb46e904b45fd52f1a12beb8b /freesurfer/models/synthstrip.1.pt /opt/freesurfer/models/synthstrip.1.pt

ENV FREESURFER_HOME=/opt/freesurfer

# Container Sentinel
ENV IS_DOCKER_8395080871=1

# Create a shared $HOME directory
RUN useradd -m -s /bin/bash -G users mriqc
WORKDIR /home/mriqc
ENV HOME="/home/mriqc"
# Refresh linked libraries
RUN ldconfig
# Installing dev requirements (packages that are not in pypi)
WORKDIR /src/
# Precaching atlases
RUN python -c "from templateflow import api as tfapi; \
	       tfapi.get('MNI152NLin2009cAsym', resolution=[1, 2], suffix=['T1w', 'T2w'], desc=None); \
	       tfapi.get('MNI152NLin2009cAsym', resolution=[1, 2], suffix='mask',\
			 desc=['brain', 'head']); \
	       tfapi.get('MNI152NLin2009cAsym', resolution=1, suffix='dseg', desc='carpet'); \
	       tfapi.get('MNI152NLin2009cAsym', resolution=1, suffix='probseg',\
			 label=['CSF', 'GM', 'WM']);\
	       tfapi.get('MNI152NLin2009cAsym', resolution=[1, 2], suffix='boldref')"

# Installing MRIQC
COPY --from=src /src/dist/*.whl .
RUN pip install --no-cache-dir $( ls *.whl )[container,rodents,test]

RUN find $HOME -type d -exec chmod go=u {} + && \
    find $HOME -type f -exec chmod go=u {} + && \
    rm -rf $HOME/.npm $HOME/.conda $HOME/.empty

# Best practices
RUN ldconfig

WORKDIR /tmp/

# Run mriqc by default
ENTRYPOINT ["/opt/conda/bin/mriqc"]
ARG BUILD_DATE
ARG VCS_REF
LABEL org.label-schema.build-date=$BUILD_DATE \
      org.label-schema.name="MRIQC" \
      org.label-schema.description="MRIQC - Automated Quality Control and visual reports for Quality Assessment of structural (T1w, T2w) and functional MRI of the brain" \
      org.label-schema.url="https://mriqc.readthedocs.io" \
      org.label-schema.vcs-ref=$VCS_REF \
      org.label-schema.vcs-url="https://github.com/nipreps/mriqc" \
      org.label-schema.version=$VERSION \
      org.label-schema.schema-version="1.0"
