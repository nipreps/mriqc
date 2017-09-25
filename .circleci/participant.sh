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

exit_docs=0
if [ "$CIRCLE_NODE_INDEX" == "1" ]; then
    mkdir -p ${SCRATCH}/docs
    docker run -i --rm=false -v ${SCRATCH}:/scratch -w /usr/local/src/mriqc/docs \
               --entrypoint=sphinx-build poldracklab/mriqc:latest -T -E -W -D language=en -b html source/ /scratch/docs 2>&1 \
               | tee ${SCRATCH}/docs/builddocs.log
    exit_docs=$( grep -qi "build succeeded." ${SCRATCH}/docs/builddocs.log; echo $? )
fi

if [ "$( grep -qiP 'docs[ _]?only' <<< "$GIT_COMMIT_MSG"; echo $?)" == "0" ]; then
    echo "Building [docs_only], nothing to do."
    exit $exit_docs
fi

MODALITY=T1w
if [ "$CIRCLE_NODE_INDEX" == "1" ]; then
    MODALITY=bold
fi

# Add a nipype config file:
mkdir -p ${HOME}/.nipype ${SCRATCH}/out/logs/nipype-${MODALITY}
echo "[execution]" > ${HOME}/.nipype/nipype.cfg
echo "profile_runtime = true" >> ${HOME}/.nipype/nipype.cfg
echo "[logging]" >> ${HOME}/.nipype/nipype.cfg
echo "log_to_file = true" >> ${HOME}/.nipype/nipype.cfg
echo "log_directory = /scratch/out/logs/nipype-${MODALITY}" >> ${HOME}/.nipype/nipype.cfg



DOCKER_RUN="docker run -i -v $HOME/data:/data:ro \
                       -v ${HOME}/.nipype:/root/.nipype \
                       -v $SCRATCH:/scratch -w /scratch \
                       ${DOCKER_IMAGE}:${DOCKER_TAG} \
                       /data/${TEST_DATA_NAME} out/ participant \
                       -v --verbose-reports --profile \
                       --n_procs 2 --ants-nthreads 1 --ants-float \
                       --webapi-url http://${MRIQC_API_HOST}/api/v1 --upload-strict"

case $CIRCLE_NODE_INDEX in
    0)
        ${DOCKER_RUN} -m T1w
        ;;
    1)
        # Run tests in bold build which is shorter
        docker run -i -v ${CIRCLE_TEST_REPORTS}:/scratch \
                   --entrypoint="py.test"  ${DOCKER_IMAGE}:${DOCKER_TAG} \
                   --ignore=src/ \
                   --junitxml=/scratch/tests.xml \
                   /usr/local/src/mriqc && \
        ${DOCKER_RUN} -m bold --testing --ica
        exit $(( $? + $exit_docs ))
        ;;
esac
