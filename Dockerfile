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

FROM poldracklab/mriqc:base

ARG PY_VER_MAJOR=3
ARG PY_VER_MINOR=5
ARG CONDA_VERSION=4.3.11

# Placeholder for niworkflows data
RUN mkdir /niworkflows_data
ENV CRN_SHARED_DATA /niworkflows_data

# Installing and setting up miniconda
RUN curl -sSLO https://repo.continuum.io/miniconda/Miniconda${PY_VER_MAJOR}-${CONDA_VERSION}-Linux-x86_64.sh && \
    bash Miniconda${PY_VER_MAJOR}-${CONDA_VERSION}-Linux-x86_64.sh -b -p /usr/local/miniconda && \
    rm Miniconda${PY_VER_MAJOR}-${CONDA_VERSION}-Linux-x86_64.sh

ENV PATH=/usr/local/miniconda/bin:$PATH \
    PYTHONNOUSERSITE=1 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    ACCEPT_INTEL_PYTHON_EULA=yes

# ENV MKL_NUM_THREADS=1 \
#     OMP_NUM_THREADS=1

# Installing precomputed python packages
RUN conda config --add channels conda-forge && \
    conda config --set always_yes yes --set changeps1 no && \
    chmod +x /usr/local/miniconda/bin/*; sync && \
    conda install -y openblas=0.2.19; sync && \
    conda install -y numpy=1.12.0 \
                     scipy=0.19.0 \
                     scikit-learn=0.18.1 \
                     matplotlib=2.0.0 \
                     pandas=0.19.2 \
                     libxml2=2.9.4 \
                     libxslt=1.1.29 \
                     statsmodels=0.8.0 \
                     dipy=0.11.0 \
                     traits=4.6.0 \
                     psutil=5.2.2 \
                     icu=58.1 \
                     scandir; \
    sync

# matplotlib cleanups: set default backend, precaching fonts
RUN sed -i 's/\(backend *: \).*$/\1Agg/g' /usr/local/miniconda/lib/python${PY_VER_MAJOR}.${PY_VER_MINOR}/site-packages/matplotlib/mpl-data/matplotlibrc && \
    python -c "from matplotlib import font_manager"


# Installing dev requirements (packages that are not in pypi)
WORKDIR /root/
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt && \
    rm -rf ~/.cache/pip

# Pre-cache niworkflows data
RUN python -c 'from niworkflows.data.getters import get_mni_icbm152_nlin_asym_09c; get_mni_icbm152_nlin_asym_09c()'

# Installing mriqc
COPY . /root/src/mriqc
RUN cd /root/src/mriqc && \
    pip install -e .[all] && \
    rm -rf ~/.cache/pip

# Metadata
ARG BUILD_DATE
ARG VCS_REF
ARG VERSION
LABEL org.label-schema.build-date=$BUILD_DATE \
      org.label-schema.name="MRIQC" \
      org.label-schema.description="MRIQC - Quality Control of structural and functional MRI" \
      org.label-schema.url="http://mriqc.readthedocs.io" \
      org.label-schema.vcs-ref=$VCS_REF \
      org.label-schema.version=$VERSION \
      org.label-schema.vcs-url="https://github.com/poldracklab/mriqc" \
      org.label-schema.schema-version="1.0"

ENTRYPOINT ["/usr/local/miniconda/bin/mriqc"]
CMD ["--help"]
