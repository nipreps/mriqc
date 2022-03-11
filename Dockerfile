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
FROM nipreps/miniconda:py38_1.4.2

ARG DEBIAN_FRONTEND=noninteractive

# Install AFNI's dependencies
RUN apt-get update \
 && apt-get install -y -q --no-install-recommends     \
                    gsl-bin                           \
                    libcurl4-openssl-dev              \
                    libgdal-dev                       \
                    libgfortran-8-dev                 \
                    libgfortran4                      \
                    libglu1-mesa-dev                  \
                    libglw1-mesa                      \
                    libgomp1                          \
                    libjpeg62                         \
                    libnode-dev                       \
                    libopenblas-dev                   \
                    libssl-dev                        \
                    libudunits2-dev                   \
                    libxm4                            \
                    libxml2-dev                       \
                    netpbm                            \
                    tcsh                              \
                    xfonts-base                       \
 && apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*                      \
 && ln -s /usr/lib/x86_64-linux-gnu/libgsl.so.23 /usr/lib/x86_64-linux-gnu/libgsl.so.19 \
 && ldconfig

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

# Installing ANTs 2.3.4 (NeuroDocker build)
ENV ANTSPATH="/opt/ants"
WORKDIR $ANTSPATH
RUN curl -sSL "https://dl.dropbox.com/s/gwf51ykkk5bifyj/ants-Linux-centos6_x86_64-v2.3.4.tar.gz" \
    | tar -xzC $ANTSPATH --strip-components 1
ENV PATH="$ANTSPATH:$PATH"

# Install FSL 5.0.11 (neurodocker build variant)
RUN echo "Downloading FSL ..." \
    && mkdir -p /opt/fsl-5.0.11 \
    && curl -fsSL --retry 5 https://fsl.fmrib.ox.ac.uk/fsldownloads/fsl-5.0.11-centos6_64.tar.gz \
    | tar -xz -C /opt/fsl-5.0.11 --strip-components 1 \
    && echo "Installing FSL conda environment ..." \
    && bash /opt/fsl-5.0.11/etc/fslconf/fslpython_install.sh -f /opt/fsl-5.0.11 \
    && rm -fr /opt/fsl-5.0.11/{doc,extras,fslpython,python,refdoc,src,tcl} \
    && rm -fr /opt/fsl-5.0.11/data/{atlases,mist,possum}

ENV FSLDIR="/opt/fsl-5.0.11" \
    PATH="/opt/fsl-5.0.11/bin:$PATH" \
    FSLOUTPUTTYPE="NIFTI_GZ" \
    FSLMULTIFILEQUIT="TRUE" \
    FSLTCLSH="/opt/fsl-5.0.11/bin/fsltclsh" \
    FSLWISH="/opt/fsl-5.0.11/bin/fslwish" \
    FSLLOCKDIR="" \
    FSLMACHINELIST="" \
    FSLREMOTECALL="" \
    FSLGECUDAQ="cuda.q" \
    POSSUMDIR="/opt/fsl-5.0.11" \
    LD_LIBRARY_PATH="/opt/fsl-5.0.11:$LD_LIBRARY_PATH"

# Unless otherwise specified each process should only use one thread - nipype
# will handle parallelization
ENV MKL_NUM_THREADS=1 \
    OMP_NUM_THREADS=1

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
COPY . /src/mriqc
ARG VERSION
# Force static versioning within container
RUN echo "${VERSION}" > /src/mriqc/mriqc/VERSION && \
    echo "include mriqc/VERSION" >> /src/mriqc/MANIFEST.in && \
    pip install --no-cache-dir "/src/mriqc[all]"

RUN find $HOME -type d -exec chmod go=u {} + && \
    find $HOME -type f -exec chmod go=u {} + && \
    rm -rf $HOME/.npm $HOME/.conda $HOME/.empty

# Best practices
RUN ldconfig
WORKDIR /tmp/
# Run mriqc by default
ENTRYPOINT ["${CONDA_PATH}/bin/mriqc"]

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
