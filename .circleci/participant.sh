#!/bin/bash
#
# Balance nipype testing workflows across CircleCI build nodes
#

# Setting      # $ help set
set -e         # Exit immediately if a command exits with a non-zero status.
set -u         # Treat unset variables as an error when substituting.
set -x         # Print command traces before executing command.


# Exit if build_only tag is found
if [ "$(grep -qiP 'build[ _]?only' <<< "$GIT_COMMIT_MSG"; echo $? )" == "0" ]; then
    exit 0
fi

ONLY_DOCS=$( grep -qviP 'docs[ _]?only' <<< "$GIT_COMMIT_MSG"; echo $?)

exit_docs=0
if [ "$CIRCLE_NODE_INDEX" == "0" ]; then
    mkdir -p ${SCRATCH}/docs
    docker run -i --rm=false -v ${SCRATCH}:/scratch -w /root/src/mriqc/docs \
               --entrypoint=sphinx-build poldracklab/mriqc:latest -T -E -W -D language=en -b html source/ /scratch/docs 2>&1 \
               | tee ${SCRATCH}/docs/builddocs.log
    cat ${SCRATCH}/docs/builddocs.log && \
    if grep -q "ERROR" ${SCRATCH}/docs/builddocs.log; then exit_docs=1; fi
fi

if [ "$ONLYDOCS" == "1" ]; then
    echo "Building [docs_only], nothing to do."
    exit $exit_docs
fi

DOCKER_RUN="docker run -i -v $HOME/data:/data:ro \
                       -v $SCRATCH:/scratch -w /scratch \
                       ${DOCKER_IMAGE}:${DOCKER_TAG} \
                       /data/${TEST_DATA_NAME} out/ participant \
                       --testing --verbose-reports --profile \
                       --n_procs 2 --ants-nthreads 1 --ica"

case $CIRCLE_NODE_INDEX in
    0)
        # Run tests in T1w build which is shorter
        docker run -i -v ${CIRCLE_TEST_REPORTS}:/scratch \
                   --entrypoint="py.test"  ${DOCKER_IMAGE}:${DOCKER_TAG} \
                   --ignore=src/ \
                   --junitxml=/scratch/tests.xml \
                   /root/src/mriqc && \
        ${DOCKER_RUN} -m T1w
        exit $(( $? + $exit_docs ))
        ;;
    1)
        ${DOCKER_RUN} -m bold
        ;;
esac
