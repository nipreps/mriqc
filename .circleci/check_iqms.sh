#!/bin/bash
#
# Balance nipype testing workflows across CircleCI build nodes
#

# Setting      # $ help set
set -e         # Exit immediately if a command exits with a non-zero status.
set -u         # Treat unset variables as an error when substituting.
set -x         # Print command traces before executing command.

if [ "$ONLYDOCS" == "1" ]; then
	echo "Building [docs_only], nothing to do."
	exit 0
fi

MODALITY=T1w
if [ "$CIRCLE_NODE_INDEX" == "1" ]; then
	MODALITY=bold
fi

echo "Checking IQMs (${MODALITY} images)..."
docker run -i -v /etc/localtime:/etc/localtime:ro \
              -v $SCRATCH:/scratch -w /scratch \
              --entrypoint="dfcheck" \
              poldracklab/mriqc:latest \
              -i /scratch/out/${MODALITY}.csv \
              -r /root/src/mriqc/mriqc/data/testdata/${MODALITY}.csv