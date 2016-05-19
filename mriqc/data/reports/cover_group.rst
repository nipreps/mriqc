
Quality Assessment - {{ modality }} group report
=========


  - Date and time: {{ timestamp }}.
  - Failed workflows: {{ failed or 'none' }}.
  - Image parameters: 
    {% for p in imparams %}- {{ imparams }}.
    {% endfor %}
    

For details on the IQMs (image quality metrics) and further information on
the enclosed plots, please read the
`user guide <http://mriqc.readthedocs.org/en/latest/userguide.html>`_.