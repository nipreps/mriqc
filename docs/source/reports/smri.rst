.. _reports-smri:

T1 and T2 -weighed images
=========================
One individual report per input structural volume will be generated
in the path ``<output_dir>/reports/sub-IDxxx_T1w.html```.
An example report is given
`here <http://web.stanford.edu/group/poldracklab/mriqc/reports/sub-51296_T1w.html>`_.

The individual report for the structural images is
structured as follows:

.. _reports-smri-summary:

Summary
-------
The first section summarizes some important information:

  * subject identifier, date and time of execution of
    ``mriqc``, software version;
  * workflow details and flags raised during execution; and
  * the extracted IQMs.

.. _reports-smri-visual:

Visual reports
--------------
The section with visual reports contains:

#. Mosaic view, zoomed-in over the parenchyma of the brain.

   .. figure:: ../resources/reports-t1w_mosaic_zoom.png
     :alt: zoomed in mosaic

#. Mosaic view with background noise enhancement.

   .. figure:: ../resources/reports-t1w_background.png
     :alt: t1 background

.. _reports-smri-verbose:

Verbose reports
---------------
If mriqc was run with the ``--verbose-reports`` flag, the
following plots will be appended:

#. Mosaic view with animation for assessment of the
   co-registration to MNI space (roll over the image
   to activate the animation).

   .. figure:: ../resources/reports-t1w_mni.svg
     :alt: t1-mni coregistration

#. Three rows of axial views at different Z-axis points
   showing: the calculated brain mask, the segmentation
   done with FSL FAST and a saturated view of the background.

   .. figure:: ../resources/reports-t1w_masks1.png
     :alt: t1 verbose masks 1

#. Two rows of coronal views, showing the object/background
   segmentation, and the air/no-air segmentation.

   .. figure:: ../resources/reports-t1w_masks2.png
     :alt: t1 verbose masks 2

#. The :math:`\chi^2` function fitting for the calculation
   of the QI2:

   .. figure:: ../resources/reports-t1w_qi2.png
     :alt: t1 fitting of QI2

.. _reports-smri-metadata:

Metadata
--------
If some metadata was found in the BIDS structure, it is
reported here.