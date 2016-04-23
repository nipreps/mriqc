
FROM oesteban/crn_nipype

ARG GIT_BRANCH
ENV GIT_BRANCH ${GIT_BRANCH:-master}

ARG GIT_URL
ENV GIT_URL ${GIT_URL:-"https://github.com/poldracklab/mriqc.git"}

WORKDIR /root/src
RUN git clone $GIT_URL && \
    cd mriqc && \
    git checkout $GIT_BRANCH && \
    source activate crnenv && \
    python setup.py install && \
    python -c "from mriqc.data import get_brainweb_1mm_normal; get_brainweb_1mm_normal()"
    
WORKDIR /root/
ADD files/run_* /usr/bin/
RUN chmod +x /usr/bin/run_*

RUN useradd -m -G users -s /bin/bash -u 1021 ubuntu
USER ubuntu

ENTRYPOINT ["/usr/bin/run_mriqc"]
CMD ["--help"]
