#!/usr/bin/env bash
# @Author: oesteban
# @Date:   2015-10-21 11:58:53
# @Last Modified by:   oesteban
# @Last Modified time: 2015-11-13 10:00:55

# Unzip bids file
tar xf ${bidsFile}

# Create output folder in scratch
mkdir -p $SCRATCH/mriqcp

# Activate anaconda env
source activate qap

# Call mriqcp
FSLOUTPUTTYPE='NIFTI_GZ' mriqcp.py -i ds003_downsampled -o $SCRATCH/mriqcp/out -w $SCRATCH/mriqcp/tmp

# Collect outputs
archivepath=$(echo $SCRATCH/mriqcp/out | cut -d: -f2 | xargs)
ln -s $archivepath .