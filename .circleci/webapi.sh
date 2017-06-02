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

MODALITY=T1w
NRECORDS=4
if [ "$CIRCLE_NODE_INDEX" == "1" ]; then
    MODALITY=bold
    NRECORDS=9
fi

echo "Checking records in MRIQC Web API"
docker run -i --entrypoint="/usr/local/miniconda/bin/python" \
              ${DOCKER_IMAGE}:${DOCKER_TAG} \
              /root/src/mriqc/mriqc/bin/mriqcwebapi_test.py \
              ${MODALITY} ${NRECORDS} \
              --webapi-url http://${MRIQC_API_HOST} --webapi-port ${MRIQC_API_PORT}