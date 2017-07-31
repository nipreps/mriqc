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

# Exit if docs_only tag is found
if [ "$(grep -qiP 'docs[ _]?only' <<< "$GIT_COMMIT_MSG"; echo $? )" == "0" ]; then
    echo "Building [docs_only], nothing to do."
    exit 0
fi

if [ "$CIRCLE_NODE_INDEX" == "0" ]; then
	# Train the classifier
	docker run -i -v $SCRATCH:/scratch -w /scratch \
	           --entrypoint="/usr/local/miniconda/bin/mriqc_clf" \
	           ${DOCKER_IMAGE}:${DOCKER_TAG} \
	           --train --test -P /usr/local/miniconda/lib/python3.6/site-packages/mriqc/data/mclf_run-20170724-191452_mod-rfc_ver-0.9.7-rc8_class-2_cv-loso_data-all_settings.yml -v
	# Run the classifier on the test data
	docker run -i -v $SCRATCH:/scratch -w /scratch \
	           --entrypoint="/usr/local/miniconda/bin/mriqc_clf" \
	           ${DOCKER_IMAGE}:${DOCKER_TAG} \
	           --load-classifier $( basename `ls $SCRATCH/mclf_run-*_class-2_cv-loso_data-all_estimator.pklz` ) -X /scratch/out/T1w.csv -v
fi
