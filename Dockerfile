
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
ADD build/files/run_* /usr/bin/
RUN chmod +x /usr/bin/run_*

ENTRYPOINT ["/usr/bin/run_mriqc"]
CMD ["--help"]
