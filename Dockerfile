FROM poldracklab/mriqc-base:latest

# Installing dev requirements (packages that are not in pypi)
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt && \
    rm -rf ~/.cache/pip

# Installing MRIQC
COPY . /root/src/mriqc
RUN cd /root/src/mriqc && pip install .[classifier,duecredit] && \
    rm -rf ~/.cache/pip

# Precaching atlases
RUN mkdir /niworkflows_data
ENV CRN_SHARED_DATA /niworkflows_data
RUN python -c 'from niworkflows.data.getters import get_mni_icbm152_nlin_asym_09c; get_mni_icbm152_nlin_asym_09c()'

ENTRYPOINT ["/usr/local/miniconda/bin/mriqc"]

ARG BUILD_DATE
ARG VCS_REF
ARG VERSION
LABEL org.label-schema.build-date=$BUILD_DATE \
      org.label-schema.name="MRIQC" \
      org.label-schema.description="MRIQC  - NR-IQMs (no-reference Image Quality Metrics) for MRI" \
      org.label-schema.url="http://mriqc.readthedocs.io" \
      org.label-schema.vcs-ref=$VCS_REF \
      org.label-schema.vcs-url="https://github.com/poldracklab/mriqc" \
      org.label-schema.version=$VERSION \
      org.label-schema.schema-version="1.0"
