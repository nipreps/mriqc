#!/bin/bash
#
# Balance nipype testing workflows across CircleCI build nodes
#

# Setting      # $ help set
set -e         # Exit immediately if a command exits with a non-zero status.
set -u         # Treat unset variables as an error when substituting.
set -x         # Print command traces before executing command.

exit_docs=0
if [ "$CIRCLE_NODE_INDEX" == "0" ]; then
	docker run -i --rm=false -v ${SCRATCH}:/scratch -w /root/src/mriqc/docs \
	           --entrypoint=sphinx-build poldracklab/mriqc:latest -T -E -W -D language=en -b html source/ /scratch/docs 2>&1 \
               | tee ${SCRATCH}/docs/builddocs.log
    cat ${SCRATCH}/docs/builddocs.log && \
    if grep -q "ERROR" ${SCRATCH}/docs/builddocs.log; then exit_docs=1; fi
fi

if [ "$ONLY_DOCS" == "1" ]; then
	echo "Building [docs_only], nothing to do."
	exit $exit_docs
fi

DOCKER_RUN="docker run -i --rm=false -v /etc/localtime:/etc/localtime:ro \
                       -v $HOME/data:/data:ro \
                       -v $SCRATCH:/scratch -w /scratch \
                       poldracklab/mriqc:latest \
                       /data/${TEST_DATA_NAME} out/ participant \
                       --verbose-reports"

case $CIRCLE_NODE_INDEX in
	0)
		# Run tests in T1w build which is shorter
		docker run -i --rm=false -v /etc/localtime:/etc/localtime:ro \
		           -v ${CIRCLE_TEST_REPORTS}:/scratch \
		           --entrypoint="py.test"  poldracklab/mriqc:latest \
		           --ignore=src/ \
		           --junitxml=/scratch/tests.xml \
		           /root/src/mriqc && \
		${DOCKER_RUN} -m T1w --testing --n_procs 1 --ants-nthreads 4
		exit $(( $? + $exit_docs ))
		;;
	1)
		${DOCKER_RUN} -m bold --ica --testing \
		              --n_procs 1 --ants-nthreads 4
		;;
esac
