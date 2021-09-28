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
FROM ubuntu:focal-20210416

# Pre-cache neurodebian key
COPY docker/files/neurodebian.gpg /usr/local/etc/neurodebian.gpg
ENV DEBIAN_FRONTEND="noninteractive" \
    LANG="C.UTF-8" \
    LC_ALL="C.UTF-8"

# Prepare environment
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
                    apt-utils \
                    autoconf \
                    build-essential \
                    bzip2 \
                    ca-certificates \
                    curl \
                    gnupg2 \
                    libtool \
                    lsb-release \
                    pkg-config \
                    tcsh \
                    xfonts-base \
                    xvfb && \
    apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Installing Neurodebian packages (FSL, AFNI, git)
RUN curl -sSL "http://neuro.debian.net/lists/$( lsb_release -c | cut -f2 ).us-ca.full" >> /etc/apt/sources.list.d/neurodebian.sources.list && \
    apt-key add /usr/local/etc/neurodebian.gpg && \
    (apt-key adv --refresh-keys --keyserver hkp://ha.pool.sks-keyservers.net 0xA5D32F012649A5A9 || true)

# Prepare environment
RUN apt-get update && \
    apt-get install -y -q --no-install-recommends \
                    bc \
                    dc \
                    ed \
                    file \
                    gsl-bin \
                    libfontconfig1 \
                    libfreetype6 \
                    libgl1-mesa-dev \
                    libgl1-mesa-dri \
                    libglib2.0-0 \
                    libglu1-mesa-dev \
                    libglw1-mesa \
                    libgomp1 \
                    libice6 \
                    libjpeg62 \
                    libtool \
                    libxcursor1 \
                    libxft2 \
                    libxinerama1 \
                    libxm4 \
                    libxrandr2 \
                    libxrender1 \
                    libxt6 \
                    netpbm \
                    wget \
    && apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && curl -sSL --retry 5 -o /tmp/multiarch.deb http://archive.ubuntu.com/ubuntu/pool/main/g/glibc/multiarch-support_2.27-3ubuntu1.2_amd64.deb \
    && dpkg -i /tmp/multiarch.deb \
    && rm /tmp/multiarch.deb \
    && curl -sSL --retry 5 -o /tmp/libxp6.deb http://mirrors.kernel.org/debian/pool/main/libx/libxp/libxp6_1.0.2-2_amd64.deb \
    && dpkg -i /tmp/libxp6.deb \
    && rm /tmp/libxp6.deb \
    && curl -sSL --retry 5 -o /tmp/libpng.deb http://snapshot.debian.org/archive/debian-security/20160113T213056Z/pool/updates/main/libp/libpng/libpng12-0_1.2.49-1%2Bdeb7u2_amd64.deb \
    && dpkg -i /tmp/libpng.deb \
    && rm /tmp/libpng.deb \
    && apt-get install -f \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && gsl2_path="$(find / -name 'libgsl.so.23' || printf '')" \
    && if [ -n "$gsl2_path" ]; then \
         ln -sfv "$gsl2_path" "$(dirname $gsl2_path)/libgsl.so.0"; \
    fi \
    && ldconfig


# Install FSL 5.0.11 (neurodocker build variant)
RUN echo "Downloading FSL ..." \
    && mkdir -p /opt/fsl-5.0.11 \
    && curl -fsSL --retry 5 https://fsl.fmrib.ox.ac.uk/fsldownloads/fsl-5.0.11-centos6_64.tar.gz \
    | tar -xz -C /opt/fsl-5.0.11 --strip-components 1 \
    && echo "Installing FSL conda environment ..." \
    && bash /opt/fsl-5.0.11/etc/fslconf/fslpython_install.sh -f /opt/fsl-5.0.11
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

# Installing and setting up miniconda
ENV CONDA_PATH="/opt/conda"
RUN curl -sSLO https://repo.continuum.io/miniconda/Miniconda3-py38_4.9.2-Linux-x86_64.sh && \
    bash Miniconda3-py38_4.9.2-Linux-x86_64.sh -b -p ${CONDA_PATH} && \
    rm Miniconda3-py38_4.9.2-Linux-x86_64.sh

# Set CPATH for packages relying on compiled libs (e.g. indexed_gzip)
ENV PATH="${CONDA_PATH}/bin:$PATH" \
    CPATH="${CONDA_PATH}/include:$CPATH" \
    PYTHONNOUSERSITE=1

# Installing conda/python environment
RUN conda install -y -c conda-forge -c anaconda \
                     python=3.8 \
                     git \
                     git-annex \
                     graphviz \
                     libxml2 \
                     libxslt \
                     matplotlib=3 \
                     mkl \
                     mkl-service \
                     nodejs \
                     numpy=1.20 \
                     pandas=1.2 \
                     pandoc=2.11 \
                     pip=21 \
                     scikit-image=0.18 \
                     scikit-learn=0.24 \
                     scipy=1.6 \
                     setuptools \
                     setuptools_scm \
                     toml \
                     traits=6.2 \
                     zlib \
                     zstd; \
                     sync && \
    chmod -R a+rX ${CONDA_PATH}; sync && \
    chmod +x ${CONDA_PATH}/bin/*; sync && \
    conda clean -y --all && sync && \
    rm -rf ~/.conda ~/.cache/pip/*; sync

# Unless otherwise specified each process should only use one thread - nipype
# will handle parallelization
ENV MKL_NUM_THREADS=1 \
    OMP_NUM_THREADS=1

# Precaching fonts, set 'Agg' as default backend for matplotlib
RUN python -c "from matplotlib import font_manager" && \
    sed -i 's/\(backend *: \).*$/\1Agg/g' $( python -c "import matplotlib; print(matplotlib.matplotlib_fname())" )

# Installing SVGO and bids-validator
RUN npm install -g svgo@^2.3 bids-validator@1.8.0 \
  && rm -rf ~/.npm ~/.empty /root/.npm

RUN pip install --no-cache-dir templateflow

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
ENTRYPOINT ["/opt/conda/bin/mriqc"]

ARG BUILD_DATE
ARG VCS_REF
LABEL org.label-schema.build-date=$BUILD_DATE \
      org.label-schema.name="MRIQC" \
      org.label-schema.description="MRIQC - Automated Quality Control and visual reports for Quality Assesment of structural (T1w, T2w) and functional MRI of the brain" \
      org.label-schema.url="http://mriqc.readthedocs.io" \
      org.label-schema.vcs-ref=$VCS_REF \
      org.label-schema.vcs-url="https://github.com/nipreps/mriqc" \
      org.label-schema.version=$VERSION \
      org.label-schema.schema-version="1.0"
