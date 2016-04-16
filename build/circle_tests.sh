#!/bin/bash

set -x
set -e

declare -a MRIQC_TESTS=(
	"-i data/ds003_downsampled -o outputs/ms-func/out -w outputs/ms-func/work --skip-anatomical" #
	"-i data/ds003_downsampled -o outputs/ms-anat/out -w outputs/ms-anat/work --skip-functional --use-plugin plugin.yml --save-memory" #
# -i data/ds003_sub-01/ -o outputs/ss-all/out -w outputs/ss-all/work
)

if [[ $CIRCLE_NODE_INDEX -eq 0 ]]; then
	docker run -i -v /etc/localtime:/etc/localtime:ro --entrypoint="/usr/bin/run_tests" oesteban/mriqc
fi

i=0
for test_params in "${MRIQC_TESTS[@]}"; do
	if [ $(($i % $CIRCLE_NODE_TOTAL)) -eq $CIRCLE_NODE_INDEX ]; then
	    docker run -i -v /etc/localtime:/etc/localtime:ro -v ~/scratch:/scratch -w /scratch oesteban/mriqc ${test_params}
	fi
	((i=i+1))
done
