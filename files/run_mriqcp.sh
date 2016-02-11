#!/bin/bash

mkdir /scratch
cd /scratch
wget "https://googledrive.com/host/0B2JWN60ZLkgkMEw4bW5VUUpSdFU/ds003_downsampled.tar"
tar xf ds003_downsampled.tar 
mriqcp.py -i ds003_downsampled -o out -w work