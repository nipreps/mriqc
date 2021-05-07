#!/bin/bash

## Script for mriqc docker container


# Command line options

#while getopts 'b' OPTION; do
 #   case "$OPTION" in
  #      b)
   #         echo "Option -b is given, converting data to BIDS format."
    #        ;;
     #   
      #  *)
       #     echo "Usage: $0 [-b] bids-root output-folder level[participant,group]" >&2
        #    exit 1
        #;;
    #esac
#done

shift "$(($OPTIND -1))"

if [ -z $1 ]; then
    echo "The BIDS input folder is not set"
    exit 1
else
    echo "BIDS folder: $1"

fi

if [ ! -z $2 ]; then
    echo "The output folder: $2"
else
    echo "The output folder is not set"
    exit 1
fi

if [ -z $3 ]; then
    echo "The subject level is not set"
    exit 1
else
    echo "Subject level: $3"

fi


# Organize Inputs
bidsdir=$1
outdir=$2
level=$3

#Check for BIDS data structure
# Install dax in DOCKERFILE



#Run MRIQC
mriqc $bidsdir $outdir $level

#Convert outputs

SUBJ=$(echo | find sub-*/ -type f | cut -d'/' -f1)
SES=$(echo | find sub-*/ -type f | cut -d'/' -f2)

export SUBJ
export SES
export outdir

/opt/xnatwrapper/html2pdf.py
/opt/xnatwrapper/json2csv.py
