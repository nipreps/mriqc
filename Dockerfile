
FROM oesteban/crn_base

WORKDIR /root
# Install nipype & mriqc
RUN mkdir -p src && \
    cd src && \
    git clone https://github.com/poldracklab/mriqc.git && \
    source activate crnenv && \
    cd mriqc && \
    pip install -e .

ADD files/run_mriqc /usr/bin/run_mriqc
RUN chmod +x /usr/bin/run_mriqc

ENTRYPOINT ["/usr/bin/run_mriqc"]
CMD ["--help"]
