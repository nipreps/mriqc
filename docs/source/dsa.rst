
.. _dsa:

Data Sharing Agreement
======================
Foundations
-----------
(Text in this section has been retrieved from https://www.ihris.org/toolkit/tools/data-sharing.html)

Data-sharing is an important way to increase the ability of researchers, scientists and policy-makers
to analyze and translate data into meaningful reports and knowledge.
Sharing data discourages duplication of effort in data collection and encourages diverse thinking,
as others are able to use the data to answer questions that the initial data collectors may not
have considered.
Sharing data also encourages accountability and transparency, enabling researchers to validate
one another's findings.
Finally, data from multiple sources can often be combined to allow for comparisons that cross national
and departmental lines.

MRIQC DSA (Data Sharing Agreement)
----------------------------------
MRIQC extracts a vector of :abbr:`IQMs (image quality metrics)` from the input dataset.
These :abbr:`IQMs (image quality metrics)` are estimations of numerical properties of the
data array in the image, hence they are not usable in identifying data.
In other words, it is not possible to identify a natural person from this information.
Additionally, MRIQC collects metadata from the appropriate fields found in the BIDS
structure of the input dataset.
Any information that could be used to identify the natural person the original data
were obtained from are stripped out (e.g. the subject identifier in the context of the
study, date of acquisition, phenotypical information, etc.).

**MRIQC does not share the original input dataset**.
Only an MD5 summary (*checksum*) of the input data is calculated and can be used to
trace back the original dataset which MRIQC extracted the information from.
This checksum cannot be used to regenerate the original data it was calculated from.
Therefore, in the case that MRIQC was run on private data, the original images
remain inaccessible to the public as no original data are shared.

Withdrawing records and period of agreement
...........................................
If you do not agree to these terms, please make use of the ``--no-sub`` (*do not submit*)
command line flag with MRIQC.
MRIQC will not collect any data when you OPT-OUT with this command line flag.
If you wish to withdraw :abbr:`IQMs (image quality metrics)` records from the database,
please send a request to the NiPreps Developers <nipreps@gmail.com> indicating the
MD5 checksums of the datasets that are to be removed.
If any MD5 checksum matches any image publicly available or one or more associated
quality rating records were found, then the corresponding record cannot be destroyed,
withdrawn or recalled.
Withdrawing can be exercised within twelve months of the submission of the IQMs to
the Web-API.
Quality ratings submitted via the "Rating Widget" of MRIQC reports or any other
means cannot be withdrawn since it requires an active action from the submitter
that makes them aware that the data will be automatically shared publicly.

The MRIQC Web-API will keep collecting records until funding runs out.
In the case the service is shut down, existing records will be posted in a permanent,
static storage service.
This agreement does not expire.

Intended use of the data
........................
The resource is particularly designed for researchers to share image quality metrics and
annotations that can readily be reused in training human experts and machine learning
algorithms.
The ultimate goal of the MRIQC Web-API is to allow the development of fully automated
quality control tools that outperform expert ratings in identifying degenerate images.

Constraints on use of the data
..............................
Data are shared under a CC0 (public domain) license.

Data confidentiality
....................
Please use please make use of the ``--no-sub`` (*do not submit*) command line flag with MRIQC
to ensure that your data remain confidential.

Financial costs of data-sharing
...............................
The MRIQC Web-API infrastructure is funded by NIMH grants R24MH117179, and ZICMH002960.
