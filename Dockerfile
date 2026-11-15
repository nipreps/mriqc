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


# AFNI, ANTs, svht_denoise and niimath on Ubuntu 22.04, for linux/amd64 and linux/arm64
# (Dockerfile.base, published by .github/workflows/docker-base.yml)
ARG BASE_IMAGE=ghcr.io/nipreps/mriqc-base:20261113-3c48cbe

#
# Build the pixi environment (pixi.lock): Python and the scientific stack from
# conda-forge, and MRIQC with its PyPI dependencies.
#
FROM ghcr.io/prefix-dev/pixi:0.82.1 AS build
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
                    ca-certificates \
                    git && \
    apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
# Run post-link scripts during install, but use global to keep out of source tree
RUN pixi config set --global run-post-link-scripts insecure

# Install dependencies before the package itself to leverage caching
WORKDIR /app
COPY pixi.lock pyproject.toml /app/
RUN --mount=type=cache,target=/root/.cache/rattler pixi install -e mriqc --frozen --skip mriqc
# Note that PATH gets hard-coded. Remove it and re-apply in final image
RUN pixi shell-hook -e mriqc --as-is | grep -v PATH > /shell-hook.sh

# Pre-fetch templates
ENV TEMPLATEFLOW_HOME="/templateflow"
RUN /app/.pixi/envs/mriqc/bin/python -c "from templateflow import api as tfapi; \
	       tfapi.get('MNI152NLin2009cAsym', resolution=[1, 2], suffix=['T1w', 'T2w'], desc=None); \
	       tfapi.get('MNI152NLin2009cAsym', resolution=[1, 2], suffix='mask',\
			 desc=['brain', 'head']); \
	       tfapi.get('MNI152NLin2009cAsym', resolution=1, suffix='dseg', desc='carpet'); \
	       tfapi.get('MNI152NLin2009cAsym', resolution=1, suffix='probseg',\
			 label=['CSF', 'GM', 'WM']);\
	       tfapi.get('MNI152NLin2009cAsym', resolution=[1, 2], suffix='boldref')"

# Finally, install the package
ARG VERSION
ENV SETUPTOOLS_SCM_PRETEND_VERSION=$VERSION
COPY . /app
RUN --mount=type=cache,target=/root/.cache/rattler pixi install -e mriqc --frozen

#
# Main stage
#
FROM ${BASE_IMAGE} AS mriqc

# Unless otherwise specified each process should only use one thread - nipype
# will handle parallelization
ENV MKL_NUM_THREADS=1 \
    OMP_NUM_THREADS=1 \
    NUMEXPR_MAX_THREADS=1

# Create a shared $HOME directory
RUN useradd -m -s /bin/bash -G users mriqc
WORKDIR /home/mriqc
ENV HOME="/home/mriqc"
# TEMPLATEFLOW_HOME is left unset on purpose: Apptainer sets $HOME to the host's home,
# which stays writable for templates the image does not ship (e.g., Fischer344).
COPY --link --from=build /templateflow /home/mriqc/.cache/templateflow

# The pixi environment, at the same path it was built in
COPY --link --from=build /app/.pixi/envs/mriqc /app/.pixi/envs/mriqc
COPY --link --from=build /shell-hook.sh /shell-hook.sh
RUN cat /shell-hook.sh >> $HOME/.bashrc
ENV PATH="/app/.pixi/envs/mriqc/bin:$PATH"

# Pacify datalad
RUN git config --global user.name "NiPreps - MRIQC" \
    && git config --global user.email "nipreps@gmail.com"

RUN chmod -R go=u $HOME

# Container Sentinel
ENV IS_DOCKER_8395080871=1

# Update version
RUN export VERSION=$(python -m mriqc --version | awk '{print $NF}') \
    && echo "VERSION=$VERSION" >> /etc/environment

WORKDIR /tmp/

# Run mriqc by default
ENTRYPOINT ["/app/.pixi/envs/mriqc/bin/mriqc"]

ARG BUILD_DATE
ARG VCS_REF
ARG VERSION
LABEL org.label-schema.build-date=$BUILD_DATE \
      org.label-schema.name="MRIQC" \
      org.label-schema.description="MRIQC - Automated Quality Control and visual reports for Quality Assessment of structural (T1w, T2w) and functional MRI of the brain" \
      org.label-schema.url="https://mriqc.readthedocs.io" \
      org.label-schema.vcs-ref=$VCS_REF \
      org.label-schema.vcs-url="https://github.com/nipreps/mriqc" \
      org.label-schema.version=$VERSION \
      org.label-schema.schema-version="1.0"
