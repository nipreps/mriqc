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

DOCKER_RUN="docker run -i -v /etc/localtime:/etc/localtime:ro \
                       -v $HOME/data:/data:ro \
                       -v $SCRATCH:/scratch -w /scratch \
                       ${DOCKER_IMAGE}:${DOCKER_TAG} \
                       /data/${TEST_DATA_NAME} out/ participant \
                       --verbose-reports"

case $CIRCLE_NODE_INDEX in
	0)
		# Run tests in T1w build which is shorter
		docker run -i -v /etc/localtime:/etc/localtime:ro \
		           -v ${CIRCLE_TEST_REPORTS}:/scratch \
		           --entrypoint="py.test"  ${DOCKER_IMAGE}:${DOCKER_TAG} \
		           --ignore=src/ \
		           --junitxml=/scratch/tests.xml \
		           /root/src/mriqc && \
		${DOCKER_RUN} -m T1w --testing --n_procs 1 --ants-nthreads 4
		;;
	1)
		${DOCKER_RUN} -m bold --ica --testing \
		              --n_procs 1 --ants-nthreads 4
		;;
esac
