#!/bin/bash

singularity run \
--cleanenv \
--contain \
--home $(pwd -P) \
--bind $(pwd -P)/BIDS/Nifti:/data \
--bind $(pwd -P)/OUTPUTS:/out \
mriqc_vuiis.simg \
--bidsdir $(pwd -P)/BIDS/Nifti \
--outdir $(pwd -P)/OUTPUTS \
--label_info "$project $subject $session $scan"