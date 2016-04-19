#!/usr/bin/env bash
# @Author: oesteban
# @Date:   2015-10-21 11:58:53
# @Last Modified by:   oesteban
# @Last Modified time: 2015-11-13 10:00:55

OUTPUT_FOLDER=$SCRATCH/${AGAVE_JOB_NAME}-${AGAVE_JOB_ID}
# Create output folder in scratch
mkdir -p ${OUTPUT_FOLDER}

echo "Running mriqc on ${bidsFile}" >> ${OUTPUT_FOLDER}/${AGAVE_JOB_NAME}.out

# Activate anaconda env
source activate crn-2.7

# Call mriqcp
FSLOUTPUTTYPE='NIFTI_GZ' mriqc -i ${bidsFile} -o ${OUTPUT_FOLDER}/out -w ${OUTPUT_FOLDER}/tmp  2>> ${OUTPUT_FOLDER}/${AGAVE_JOB_NAME}.err 1>> ${OUTPUT_FOLDER}/${AGAVE_JOB_NAME}.out

# Collect outputs
archivepath=$(echo ${OUTPUT_FOLDER}/out | cut -d: -f2 | xargs)
ln -s $archivepath .