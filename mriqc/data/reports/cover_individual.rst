{{ title }}


  - Subject ID: {{ sub_id }}.
  - Date and time: {{ timestamp }}.
  - MRIQC version: {{ version }}.
  - Sessions and (scans) failed: {{ failed or 'none'}}.
  - Image parameters:

{{ imparams }}


For details on the IQMs (image quality metrics) and further information on
the enclosed plots, please read the
`user guide <http://mriqc.readthedocs.org/en/latest/userguide.html>`_.