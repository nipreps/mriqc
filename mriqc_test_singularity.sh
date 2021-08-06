#!/bin/bash

project='project1'
subject='subject1'
session='session1'
scan='scan1'


singularity run \
--cleanenv \
--contain \
--home $(pwd -P) \
--bind $(pwd -P)/BIDS/Nifti:/data \
--bind $(pwd -P)/OUTPUTS:/out \
mriqc_v4.simg \
--bidsdir $(pwd -P)/BIDS/Nifti \
--outdir $(pwd -P)/OUTPUTS_4 \
--label_info "$project $subject $session $scan"