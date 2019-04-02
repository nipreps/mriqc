# Use Ubuntu 16.04 LTS
FROM ubuntu:xenial-20161213

# Pre-cache neurodebian key
COPY docker/files/neurodebian.gpg /usr/local/etc/neurodebian.gpg

# Prepare environment
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
                    autoconf \
                    build-essential \
                    bzip2 \
                    ca-certificates \
                    curl \
                    cython3 \
                    git \
                    graphviz=2.38.0-12ubuntu2 \
                    libtool \
                    pkg-config \
                    xvfb && \
    curl -sL https://deb.nodesource.com/setup_10.x | bash - && \
    apt-get install -y --no-install-recommends \
                    nodejs && \
    apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Installing Neurodebian packages (FSL, AFNI, git)
RUN curl -sSL "http://neuro.debian.net/lists/$( lsb_release -c | cut -f2 ).us-ca.full" >> /etc/apt/sources.list.d/neurodebian.sources.list && \
    apt-key add /usr/local/etc/neurodebian.gpg && \
    (apt-key adv --refresh-keys --keyserver hkp://ha.pool.sks-keyservers.net 0xA5D32F012649A5A9 || true)

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
                    fsl-core=5.0.9-5~nd16.04+1 \
                    afni=16.2.07~dfsg.1-5~nd16.04+1 \
                    convert3d \
                    git-annex-standalone && \
    apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

ENV FSLDIR="/usr/share/fsl/5.0" \
    FSLOUTPUTTYPE="NIFTI_GZ" \
    FSLMULTIFILEQUIT="TRUE" \
    POSSUMDIR="/usr/share/fsl/5.0" \
    LD_LIBRARY_PATH="/usr/lib/fsl/5.0:$LD_LIBRARY_PATH" \
    FSLTCLSH="/usr/bin/tclsh" \
    FSLWISH="/usr/bin/wish" \
    AFNI_MODELPATH="/usr/lib/afni/models" \
    AFNI_IMSAVE_WARNINGS="NO" \
    AFNI_TTATLAS_DATASET="/usr/share/afni/atlases" \
    AFNI_PLUGINPATH="/usr/lib/afni/plugins"
ENV PATH="/usr/lib/fsl/5.0:/usr/lib/afni/bin:$PATH"

# Installing ANTs 2.2.0 (NeuroDocker build)
ENV ANTSPATH=/usr/lib/ants
RUN mkdir -p $ANTSPATH && \
    curl -sSL "https://dl.dropbox.com/s/2f4sui1z6lcgyek/ANTs-Linux-centos5_x86_64-v2.2.0-0740f91.tar.gz" \
    | tar -xzC $ANTSPATH --strip-components 1
ENV PATH=$ANTSPATH:$PATH

# Create a shared $HOME directory
RUN useradd -m -s /bin/bash -G users bidsapp
WORKDIR /home/bidsapp
ENV HOME="/home/bidsapp"

# Installing SVGO
RUN curl -sL https://deb.nodesource.com/setup_10.x | bash -
RUN apt-get install -y nodejs
RUN npm install -g svgo

# Installing and setting up miniconda
RUN curl -sSLO https://repo.continuum.io/miniconda/Miniconda3-4.5.11-Linux-x86_64.sh && \
    bash Miniconda3-4.5.11-Linux-x86_64.sh -b -p /usr/local/miniconda && \
    rm Miniconda3-4.5.11-Linux-x86_64.sh

# Set CPATH for packages relying on compiled libs (e.g. indexed_gzip)
ENV PATH="/usr/local/miniconda/bin:$PATH" \
    CPATH="/usr/local/miniconda/include/:$CPATH" \
    LANG="C.UTF-8" \
    LC_ALL="C.UTF-8" \
    PYTHONNOUSERSITE=1

# Installing precomputed python packages
RUN conda install -y python=3.7.1 \
                     mkl=2018.0.3 \
                     mkl-service \
                     numpy=1.15.4 \
                     scipy=1.1.0 \
                     scikit-learn=0.19.1 \
                     matplotlib=2.2.2 \
                     pandas=0.23.4 \
                     libxml2=2.9.8 \
                     libxslt=1.1.32 \
                     graphviz=2.40.1 \
                     traits=4.6.0 \
                     zlib; sync && \
    chmod -R a+rX /usr/local/miniconda; sync && \
    chmod +x /usr/local/miniconda/bin/*; sync && \
    conda build purge-all; sync && \
    conda clean -tipsy && sync

# Unless otherwise specified each process should only use one thread - nipype
# will handle parallelization
ENV MKL_NUM_THREADS=1 \
    OMP_NUM_THREADS=1

# Precaching fonts, set 'Agg' as default backend for matplotlib
RUN python -c "from matplotlib import font_manager" && \
    sed -i 's/\(backend *: \).*$/\1Agg/g' $( python -c "import matplotlib; print(matplotlib.matplotlib_fname())" )

# Precaching atlases
ENV TEMPLATEFLOW_HOME="/opt/templateflow"
RUN mkdir -p $TEMPLATEFLOW_HOME
RUN pip install --no-cache-dir "templateflow>=0.1.3,<0.2.0a0" && \
    python -c "from templateflow import api as tfapi; \
               tfapi.get('MNI152NLin2009cAsym'); \
               tfapi.get('OASIS30ANTs');"

# Installing dev requirements (packages that are not in pypi)
WORKDIR /src/
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Installing MRIQC
COPY . /src/mriqc
ARG VERSION
# Force static versioning within container
RUN echo "${VERSION}" > /src/mriqc/mriqc/VERSION && \
    echo "include mriqc/VERSION" >> /src/mriqc/MANIFEST.in && \
    cd /src/mriqc && \
    pip install --no-cache-dir .[all]

RUN find $HOME -type d -exec chmod go=u {} + && \
    find $HOME -type f -exec chmod go=u {} +


# Best practices
RUN ldconfig
WORKDIR /tmp/

# Run mriqc by default
ENTRYPOINT ["/usr/local/miniconda/bin/mriqc"]

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
