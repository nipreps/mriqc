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

# Use Ubuntu 20.04 LTS
FROM nipreps/miniconda:py39_2205.0

ARG DEBIAN_FRONTEND=noninteractive
ENV LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:${CONDA_PATH}/lib"

# Install AFNI latest (neurodocker build)
ENV AFNI_DIR="/opt/afni"
RUN echo "Downloading AFNI ..." \
    && mkdir -p ${AFNI_DIR} \
    && curl -fsSL --retry 5 https://afni.nimh.nih.gov/pub/dist/tgz/linux_openmp_64.tgz \
    | tar -xz -C ${AFNI_DIR} --strip-components 1
ENV PATH="${AFNI_DIR}:$PATH" \
    AFNI_IMSAVE_WARNINGS="NO" \
    AFNI_MODELPATH="${AFNI_DIR}/models" \
    AFNI_TTATLAS_DATASET="${AFNI_DIR}/atlases" \
    AFNI_PLUGINPATH="${AFNI_DIR}/plugins"

# Install AFNI's dependencies
RUN ${CONDA_PATH}/bin/mamba install -c conda-forge -c anaconda \
                            gsl                                \
                            xorg-libxp                         \
                            scipy=1.8                          \
    && ${CONDA_PATH}/bin/mamba install -c sssdgc png \
    && sync \
    && ${CONDA_PATH}/bin/conda clean -afy; sync \
    && rm -rf ~/.conda ~/.cache/pip/*; sync \
    && ln -s ${CONDA_PATH}/lib/libgsl.so.25 /usr/lib/x86_64-linux-gnu/libgsl.so.19 \
    && ln -s ${CONDA_PATH}/lib/libgsl.so.25 /usr/lib/x86_64-linux-gnu/libgsl.so.0 \
    && ldconfig

RUN apt-get update \
 && apt-get install -y -q --no-install-recommends     \
                    libcurl4-openssl-dev              \
                    libgdal-dev                       \
                    libgfortran-8-dev                 \
                    libgfortran4                      \
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

# Installing ANTs 2.3.4 (NeuroDocker build)
ENV ANTSPATH="/opt/ants"
WORKDIR $ANTSPATH
RUN curl -sSL "https://dl.dropbox.com/s/gwf51ykkk5bifyj/ants-Linux-centos6_x86_64-v2.3.4.tar.gz" \
    | tar -xzC $ANTSPATH --strip-components 1
ENV PATH="$ANTSPATH:$PATH"

# Install FSL 5.0.11
RUN curl -sSL https://fsl.fmrib.ox.ac.uk/fsldownloads/fsl-5.0.11-centos7_64.tar.gz | tar zxv --no-same-owner -C /opt \
    --exclude='fsl/doc' \
    --exclude='fsl/refdoc' \
    --exclude='fsl/python/oxford_asl' \
    --exclude='fsl/data/possum' \
    --exclude='fsl/data/first' \
    --exclude='fsl/data/mist' \
    --exclude='fsl/data/atlases' \
    --exclude='fsl/data/xtract_data' \
    --exclude='fsl/extras/doc' \
    --exclude='fsl/extras/man' \
    --exclude='fsl/extras/src' \
    --exclude='fsl/src' \
    --exclude='fsl/tcl'

ENV FSLDIR="/opt/fsl" \
    PATH="/opt/fsl/bin:$PATH" \
    FSLOUTPUTTYPE="NIFTI_GZ" \
    FSLMULTIFILEQUIT="TRUE" \
    FSLTCLSH="/opt/fsl/bin/fsltclsh" \
    FSLWISH="/opt/fsl/bin/fslwish" \
    FSLLOCKDIR="" \
    FSLMACHINELIST="" \
    FSLREMOTECALL="" \
    FSLGECUDAQ="cuda.q" \
    POSSUMDIR="/opt/fsl" \
    LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/opt/fsl"

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

RUN git config --global user.name "NiPrep MRIQC" \
    && git config --global user.email "nipreps@gmail.com"
# Installing MRIQC
COPY . /src/mriqc
# Force static versioning within container
ARG VERSION


RUN export SETUPTOOLS_SCM_PRETEND_VERSION=$VERSION && \
    pip install --no-cache-dir "/src/mriqc[all]"

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
      org.label-schema.url="http://mriqc.readthedocs.io" \
      org.label-schema.vcs-ref=$VCS_REF \
      org.label-schema.vcs-url="https://github.com/nipreps/mriqc" \
      org.label-schema.version=$VERSION \
      org.label-schema.schema-version="1.0"
