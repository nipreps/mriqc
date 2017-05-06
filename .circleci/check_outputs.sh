#!/bin/bash
#
# Balance nipype testing workflows across CircleCI build nodes
#

# Setting      # $ help set
set -e         # Exit immediately if a command exits with a non-zero status.
set -u         # Treat unset variables as an error when substituting.

# Exit if build_only tag is found
if [ "$(grep -qiP 'build[ _]?only' <<< "$GIT_COMMIT_MSG"; echo $? )" == "0" ]; then
	exit 0
fi

MODALITY=T1w
if [ "$CIRCLE_NODE_INDEX" == "1" ]; then
	MODALITY=bold
fi

echo "Checking outputs (${MODALITY})..."
find $SCRATCH/out/   | sed s+$SCRATCH/++ | sort > $SCRATCH/outputs.txt
diff $HOME/$CIRCLE_PROJECT_REPONAME/tests/circle_${MODALITY}.txt $SCRATCH/outputs.txt
exit_code=$?

echo "Checking nifti files (${MODALITY})..."
# Have docker run cmd handy
HASHCMD="docker run -i -v $SCRATCH:/scratch \
                    --entrypoint=/usr/local/miniconda/bin/nib-hash \
                    poldracklab/mriqc:latest"


find $SCRATCH -name "*.nii.gz" -type f  | sed s+$SCRATCH+/scratch+ | sort | xargs -n1 $HASHCMD >> $SCRATCH/nii_outputs.txt
diff $HOME/$CIRCLE_PROJECT_REPONAME/tests/nii_${MODALITY}.txt $SCRATCH/nii_outputs.txt
exit $(( $? + $exit_code ))