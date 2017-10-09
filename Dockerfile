# Use Ubuntu 16.04 LTS
FROM ubuntu:xenial-20161213

# Pre-cache neurodebian key
COPY docker/files/neurodebian.gpg /root/.neurodebian.gpg

# Prepare environment
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
                    curl \
                    bzip2 \
                    ca-certificates \
                    xvfb  \
                    cython3 \
                    build-essential \
                    autoconf \
                    libtool \
                    pkg-config && \
    curl -sSL http://neuro.debian.net/lists/xenial.us-ca.full >> /etc/apt/sources.list.d/neurodebian.sources.list && \
    apt-key add /root/.neurodebian.gpg && \
    (apt-key adv --refresh-keys --keyserver hkp://ha.pool.sks-keyservers.net 0xA5D32F012649A5A9 || true)

# Installing Neurodebian packages (FSL, AFNI, git)
RUN apt-get update  && \
    apt-get install -y --no-install-recommends \
                    fsl-core \
                    fsl-mni152-templates \
                    afni

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
ENV ANTSPATH=/opt/ants \
    PATH=/opt/ants:$PATH

# Installing WEBP tools
RUN curl -sSLO "http://downloads.webmproject.org/releases/webp/libwebp-0.5.2-linux-x86-64.tar.gz" && \
    tar -xf libwebp-0.5.2-linux-x86-64.tar.gz && cd libwebp-0.5.2-linux-x86-64/bin && \
    mv cwebp /usr/local/bin/ && rm -rf libwebp-0.5.2-linux-x86-64

# Installing SVGO
RUN curl -sL https://deb.nodesource.com/setup_7.x | bash -
RUN apt-get install -y nodejs
RUN npm install -g svgo

# Installing Ubuntu packages and cleaning up
RUN apt-get install -y --no-install-recommends \
                    git=1:2.7.4-0ubuntu1 \
                    graphviz=2.38.0-12ubuntu2 && \
    apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Installing and setting up miniconda
RUN curl -sSLO https://repo.continuum.io/miniconda/Miniconda3-4.3.11-Linux-x86_64.sh && \
    bash Miniconda3-4.3.11-Linux-x86_64.sh -b -p /usr/local/miniconda && \
    rm Miniconda3-4.3.11-Linux-x86_64.sh

ENV PATH=/usr/local/miniconda/bin:$PATH \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

# Installing precomputed python packages
RUN conda install -c conda-forge -y openblas=0.2.19; \
    sync && \
    conda install -c conda-forge -y \
                     numpy=1.12.0 \
                     cython \
                     scipy=0.19.0 \
                     matplotlib=2.0.0 \
                     pandas=0.19.2 \
                     libxml2=2.9.4 \
                     libxslt=1.1.29 \
                     sympy=1.0 \
                     statsmodels=0.8.0 \
                     dipy=0.11.0 \
                     traits=4.6.0 \
                     psutil=5.2.2 \
                     sphinx=1.5.4; \
    sync &&  \
    chmod -R a+rX /usr/local/miniconda && \
    chmod +x /usr/local/miniconda/bin/* && \
    conda clean --all -y; sync && \
    python -c "from matplotlib import font_manager" && \
    sed -i 's/\(backend *: \).*$/\1Agg/g' $( python -c "import matplotlib; print(matplotlib.matplotlib_fname())" )

# Unless otherwise specified each process should only use one thread - nipype
# will handle parallelization
ENV MKL_NUM_THREADS=1 \
    OMP_NUM_THREADS=1

# Installing dev requirements (packages that are not in pypi)
WORKDIR /usr/local/src/mriqc
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt && \
    rm -rf ~/.cache/pip

# Precaching atlases after niworkflows is available
RUN mkdir /niworkflows_data
ENV CRN_SHARED_DATA /niworkflows_data
RUN python -c 'from niworkflows.data.getters import get_mni_icbm152_nlin_asym_09c; get_mni_icbm152_nlin_asym_09c()'

# Installing MRIQC
COPY . /usr/local/src/mriqc
ARG VERSION
RUN echo "${VERSION}" > mriqc/VERSION && \
    pip install .[all] && \
    rm -rf ~/.cache/pip

RUN ldconfig

ENTRYPOINT ["/usr/local/miniconda/bin/mriqc"]

# Store metadata
ARG BUILD_DATE
ARG VCS_REF
LABEL org.label-schema.build-date=$BUILD_DATE \
      org.label-schema.name="MRIQC" \
      org.label-schema.description="MRIQC - Automated Quality Control and visual reports for Quality Assesment of structural (T1w, T2w) and functional MRI of the brain" \
      org.label-schema.url="http://mriqc.readthedocs.io" \
      org.label-schema.vcs-ref=$VCS_REF \
      org.label-schema.vcs-url="https://github.com/poldracklab/mriqc" \
      org.label-schema.version=$VERSION \
      org.label-schema.schema-version="1.0"
