
.. _smri:


T1 and T2 -weighed images
-------------------------

After all processing has been completed, the designated output directory for the ``mriqc`` workflow will contain a set of pdf
files that contain the relevant reports for the set of scans undergoing
quality assessment. The set of output pdfs includes one pdf file per
input scan in the scan directory, e.g.:
``T1w_sub-01.html``, which contains the T1 slice
mosaic and :abbr:`IQMs (image quality metrics)` for that scan. There will also be a group report
pdf in the main output directory, e.g.:
``T1w_group.html``, that contains summary metrics for
the entire set of scans.


For the individual scan reports:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**The T1 Anatomical Slice Mosaic**:
This plot in the report for the scan being assessed, e.g.:

.. figure:: resources/reports-anatomical-mosaic.png
  :alt: example of mosaic view of one structural MRI image

  is the rendering of the axial slices from the 3D stack created
  by the workflow.

This image can be used to eyeball the quality of the overall
signal in the anatomical scan, as it will be obvious if there were any
problem areas where there was signal dropout resulting from a bad shim
or other sources of signal distortion.

**Metrics**: The :abbr:`IQMs (image quality metrics)` displayed in the Summary Report, e.g.:

.. figure:: resources/reports-anatomical-violin.png
  :alt: example of mosaic view of one structural MRI image

  The stars in these plots denote where the score for the scans for
  this participant fall in the distribution of all scores for scans that
  were included as inputs to the anatomical-spatial workflow. If there are
  several runs per session for this individual, then the stars will be
  displayed adjacent to each other in the violin plot.


For the group reports:
^^^^^^^^^^^^^^^^^^^^^^

The violin plots included in the group report, e.g.:
``anatomical_group.html``, are a graphical representation of
the columnar values in the ``aMRIQC.csv`` file that was
created in the main output directory for the workflow. The scores for
each metric described above were aggregated to create the distributions
that were plotted in both the individual and group reports. Hence, the
violin plots in the individual scan reports and the group reports are
identical, except that the group reports do not contain any stars
denoting individual scans. These group reports are intended to provide
the user a means of visually inspecting the overall quality of the
spatial data for that group of anatomical scans.