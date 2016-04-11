
FROM oesteban/crn_nipype

WORKDIR /root
# Install nipype & mriqc
RUN mkdir -p src && \
    cd src && \
    git clone https://github.com/poldracklab/mriqc.git && \
    source activate crnenv && \
    cd mriqc && \
    pip install -e .

ADD files/run_mriqc /usr/bin/run_mriqc
ADD files/run_tests /usr/bin/run_tests
RUN chmod +x /usr/bin/run_mriqc && \
	chmod +x /usr/bin/run_tests

ENTRYPOINT ["/usr/bin/run_mriqc"]
CMD ["--help"]
