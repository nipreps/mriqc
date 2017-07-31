
===================================
The MRIQC classifier for T1w images
===================================

MRIQC is released with two classifiers (already trained) to predict image quality
of T1w images.

From our preprint `MRIQC: Advancing the Automatic Prediction of Image Quality in MRI from Unseen Sites
<https://doi.org/10.1101/111294>`_:

    *Quality control of MRI is essential for excluding problematic 
    acquisitions and avoiding bias in subsequent image processing and analysis.
    Visual inspection is subjective and impractical for large scale datasets. 
    Although automated quality assessments have been demonstrated on single-site datasets, 
    it is unclear that solutions can generalize to unseen data acquired at new sites. 
    Here, we introduce the MRI Quality Control tool (MRIQC), a tool for extracting 
    quality measures and fitting a binary (accept/exclude) classifier. 
    The classifier is trained on a publicly available, multi-site dataset (17 sites, N=1102). 
    We perform model selection evaluating different normalization and feature exclusion
    approaches aimed at maximizing across-site generalization and estimate an accuracy 
    of 76%±13% on new sites, using leave-one-site-out cross-validation.
    We confirm that result on a held-out dataset (2 sites, N=265) also obtaining 
    a 76% accuracy. 
    Even though the performance of the trained classifier is statistically above chance, 
    we show that it is susceptible to site effects and unable to account for artifacts 
    specific to new sites. MRIQC performs with high accuracy in intra-site prediction, 
    but performance on unseen sites leaves space for improvement which might require 
    more labeled data and new approaches to the between-site variability.
    Overcoming these limitations is crucial for a more objective quality assessment 
    of neuroimaging data, and to enable the analysis of extremely large and 
    multi-site samples.*

.. automodule:: mriqc.classifier
    :members:
    :undoc-members:
    :show-inheritance:

