#!/bin/bash

### Script for mriqc docker container
# Dylan Lawless


# Initialize defaults
export bidsdir=NO_BIDS
export outdir=NO_OUTDIR
export level=participant

# Parse options
while [[ $# -gt 0 ]]; do
  key="${1}"
  case $key in
    --bidsdir)
      export bidsdir="${2}"; shift; shift ;;
    --outdir)
      export outdir="${2}"; shift; shift ;;
    *)
      echo Unknown input "${1}"; shift ;;
  esac
done

#Run MRIQC
mriqc --no-sub -v ${bidsdir} ${outdir} ${level} 

#Convert outputs
cd ${outdir}

#Run py scripts to convert outputs
/opt/xnatwrapper/convert_outputs.py
