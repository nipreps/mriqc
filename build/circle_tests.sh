#!/bin/bash

set -x
set -e

declare -a MRIQC_TESTS=(
	"-i data/ds003_downsampled -o outputs/ms-func/out -w outputs/ms-func/work --skip-anatomical" #
	"-i data/ds003_downsampled -o outputs/ms-anat/out -w outputs/ms-anat/work --skip-functional --use-plugin plugin.yml --save-memory" #
# -i data/ds003_sub-01/ -o outputs/ss-all/out -w outputs/ss-all/work
)

i=0
for test_params in "${MRIQC_TESTS[@]}"; do
	if [ $($i) -eq 0 ]; then
		docker run -i -e TZ=PST --entrypoint="/usr/bin/run_tests" oesteban/mriqc
	fi
	if [ $(($i % $CIRCLE_NODE_TOTAL)) -eq $CIRCLE_NODE_INDEX ]; then
	    docker run -i -e TZ=PST -v ~/scratch:/scratch -w /scratch oesteban/mriqc ${test_params}
	fi
	((i=i+1))
done
