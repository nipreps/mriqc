
FROM oesteban/crn_nipype

WORKDIR /root/src
ADD . mriqc/
# Install nipype & mriqc
RUN source activate crnenv && \
    pip install --upgrade numpy && \
    cd mriqc && \
    pip install -e . && \
    python -c "from matplotlib import font_manager" && \
    python -c "from mriqc.data import get_brainweb_1mm_normal; get_brainweb_1mm_normal()"

WORKDIR /root/
ADD build/files/run_mriqc /usr/bin/run_mriqc
ADD build/files/run_tests /usr/bin/run_tests
ADD build/files/run_dfcheck /usr/bin/run_dfcheck

RUN chmod +x /usr/bin/run_mriqc && \
	chmod +x /usr/bin/run_tests && \
	chmod +x /usr/bin/run_dfcheck

ENTRYPOINT ["/usr/bin/run_mriqc"]
CMD ["--help"]
