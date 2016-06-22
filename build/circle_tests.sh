#!/bin/bash

set -x
set -e

# declare -a MRIQC_TESTS=(
# 	"-i data/ds003_downsampled -o outputs/ms-func/out -w outputs/ms-func/work --skip-anatomical" #
# 	"-i data/ds003_downsampled -o outputs/ms-anat/out -w outputs/ms-anat/work --skip-functional --use-plugin plugin.yml --save-memory" #
# # -i data/ds003_sub-01/ -o outputs/ss-all/out -w outputs/ss-all/work
# )


# i=0
# for test_params in "${MRIQC_TESTS[@]}"; do
# 	if [ $(($i % $CIRCLE_NODE_TOTAL)) -eq $CIRCLE_NODE_INDEX ]; then
# 	    docker run -i -v /etc/localtime:/etc/localtime:ro -v ~/scratch:/scratch -w /scratch oesteban/mriqc ${test_params}
# 	fi
# 	((i=i+1))
# done
cp build/ants_settings_test.json ~/scratch/data

if [[ $CIRCLE_NODE_INDEX -eq 0 ]]; then
	docker run -i -v /etc/localtime:/etc/localtime:ro -v ~/scratch:/scratch -w /scratch oesteban/mriqc -B /scratch/data/ds003_downsampled -d func -o outputs/ms-func/out -w outputs/ms-func/work
	docker run -i -v /etc/localtime:/etc/localtime:ro -v ~/scratch:/scratch -w /scratch oesteban/mriqc -B /scratch/data/ds003_downsampled -S $(seq -f '%02g' 01 05) -d anat -o outputs/ms-anat/out -w outputs/ms-anat/work --ants-settings /scratch/data/ants_settings_test.json
	docker run -i -v /etc/localtime:/etc/localtime:ro -v ~/scratch:/scratch -w /scratch --entrypoint="/usr/bin/run_mriqc_plot" oesteban/mriqc -d anat -o outputs/ms-anat/out -w outputs/ms-anat/work
fi

if [[ $CIRCLE_NODE_INDEX -eq 1 ]]; then
	docker run -i -v /etc/localtime:/etc/localtime:ro --entrypoint="/usr/bin/run_tests" oesteban/mriqc
	docker run -i -v /etc/localtime:/etc/localtime:ro -v ~/scratch:/scratch -w /scratch oesteban/mriqc -B /scratch/data/ds003_downsampled -S $(seq -f '%02g' 06 13) -d anat -o outputs/ms-anat/out -w outputs/ms-anat/work --ants-settings /scratch/data/ants_settings_test.json
	docker run -i -v /etc/localtime:/etc/localtime:ro -v ~/scratch:/scratch -w /scratch --entrypoint="/usr/bin/run_mriqc_plot" oesteban/mriqc -d anat -o outputs/ms-anat/out -w outputs/ms-anat/work
fi