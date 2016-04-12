
FROM oesteban/crn_nipype

ARG GIT_BRANCH
ENV GIT_BRANCH ${GIT_BRANCH:-master}

ARG GIT_URL
ENV GIT_URL ${GIT_URL:-"https://github.com/poldracklab/mriqc.git"}

WORKDIR /root
# Install nipype & mriqc
RUN mkdir -p src && \
    cd src && \
    git clone ${GIT_URL} && \
    source activate crnenv && \
    cd mriqc && \
    git pull && \
    git checkout ${GIT_BRANCH} && \
    pip install -e .

ADD files/run_mriqc /usr/bin/run_mriqc
ADD files/run_tests /usr/bin/run_tests
RUN chmod +x /usr/bin/run_mriqc && \
	chmod +x /usr/bin/run_tests

ENTRYPOINT ["/usr/bin/run_mriqc"]
CMD ["--help"]
