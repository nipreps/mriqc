FROM oesteban/crn_nipype_py3:latest

ADD build/files/run_* /usr/bin/
RUN chmod +x /usr/bin/run_*

WORKDIR /root/src

COPY . mriqc/

# Install nipype & mriqc
RUN source activate crnenv && \
    cd mriqc && \
    rm -fr /root/miniconda/envs/crnenv/lib/python3.5/site-packages/rst2pdf* && \
    pip install -e git+https://github.com/oesteban/rst2pdf.git@futurize/stage2#egg=rst2pdf && \
    pip install -r requirements.txt && \
    pip install -e . && \
    python -c "from matplotlib import font_manager" && \
    python -c "from mriqc.data import get_brainweb_1mm_normal; get_brainweb_1mm_normal()"

COPY build/files/run_* /usr/bin/
RUN chmod +x /usr/bin/run_*

WORKDIR /scratch
ENTRYPOINT ["/usr/bin/run_mriqc"]
CMD ["--help"]
