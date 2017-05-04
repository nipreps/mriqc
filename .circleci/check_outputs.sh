#!/bin/bash
#
# Balance nipype testing workflows across CircleCI build nodes
#

# Setting      # $ help set
set -e         # Exit immediately if a command exits with a non-zero status.
set -u         # Treat unset variables as an error when substituting.
set -x         # Print command traces before executing command.

MODALITY=T1w
if [ "$CIRCLE_NODE_INDEX" == "1" ]; then
	MODALITY=bold
fi

echo "Checking outputs (${MODALITY} images)..."

cd $SCRATCH
find out/ | sort > $SCRATCH/outputs.txt

diff $HOME/$CIRCLE_PROJECT_REPONAME/tests/circle_${MODALITY}.txt $SCRATCH/outputs.txt
