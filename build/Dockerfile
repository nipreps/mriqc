
FROM oesteban/crn_nipype

ARG GIT_BRANCH
ENV GIT_BRANCH ${GIT_BRANCH:-master}

ARG GIT_URL
ENV GIT_URL ${GIT_URL:-"https://github.com/poldracklab/mriqc.git"}

WORKDIR /root/src

# Install nipype & mriqc
RUN source activate crnenv && \
    pip install --upgrade numpy && \
    python -c "from matplotlib import font_manager"

RUN git clone $GIT_URL && \
    cd mriqc && \
    git checkout $GIT_BRANCH && \
    source activate crnenv && \
    pip install -e . && \
    python -c "from mriqc.data import get_brainweb_1mm_normal; get_brainweb_1mm_normal()"

WORKDIR /root/
ADD files/run_mriqc /usr/bin/run_mriqc
ADD files/run_tests /usr/bin/run_tests
ADD files/run_dfcheck /usr/bin/run_dfcheck
RUN chmod +x /usr/bin/run_mriqc && \
	chmod +x /usr/bin/run_tests && \
	chmod +x /usr/bin/run_dfcheck

ENTRYPOINT ["/usr/bin/run_mriqc"]
CMD ["--help"]
