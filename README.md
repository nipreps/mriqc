# mriqc
Quality Control Protocols (aka QAP) for MRI


## Dependencies
Make sure you have FSL and AFNI installed, and the binaries available
in the system's $PATH.

## Installation
Just issue:

```
pip install -e git+https://github.com/poldracklab/mriqc.git#egg=mriqc
```

## Example command line:

```
mriqc -i ~/Data/ds003_downsampled -o out/ -w work/
```
