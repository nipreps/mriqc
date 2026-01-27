25.0.0 (TBD)
============
A new 25.x series.

CHANGES
-------
  * FIX: Decode bytes to produce a string (#1371)
  * FIX: Containers report wrong version (#1357)
  * FIX: Elapsed run time format (#1366)
  * FIX: Passing ``bytes`` into ``str.join()`` (#1356)
  * FIX: Optimize interface to minimize memory fingerprint (#1351)
  * ENH: Remove redundant *DIPY* function (#1369)
  * DOC: Add a HPC troubleshooting section to the documentation (#1349)
  * DOC: Provide a range of years ending with current for the copyright statement (#1359)
  * DOC: Fix missing code block start (#1368)
  * DOC: Reorganize documentation and redirect to *NiPreps* docs (#1367)

24.0.2 (August 26, 2024)
========================
A patch release with bugfixes and enhancements.

CHANGES
-------

* FIX: Pin latest *NiReports* release (24.0.2) addressing ``fMRIPlot`` issues by @oesteban (`#1342 <https://github.com/nipreps/mriqc/pull/1342>`__)
* FIX: Edge artifacts in first and last slices due to interpolation by @oesteban (`#1338 <https://github.com/nipreps/mriqc/pull/1338>`__)
* FIX: Normalize bids-filters' modality keys to be lowercase by @oesteban (`#1332 <https://github.com/nipreps/mriqc/pull/1332>`__)
* ENH: Add license NOTICE to start banner by @oesteban (`#1343 <https://github.com/nipreps/mriqc/pull/1343>`__)
* ENH: Enable writing crashfiles in compressed-pickle format by @oesteban (`#1339 <https://github.com/nipreps/mriqc/pull/1339>`__)
* ENH: Use ``orjson`` to serialize JSON, addressing *Numpy* serialization issues by @oesteban (`#1337 <https://github.com/nipreps/mriqc/pull/1337>`__)
* ENH: Handle WebAPI timeouts more gently by @oesteban (`#1336 <https://github.com/nipreps/mriqc/pull/1336>`__)


24.0.1 (August 20, 2024)
========================
A patch release with a large number of bugfixes (mostly focusing on memory issues), maintenance
activities, and metadata crawling before *Nipype* kicks in as a major optimization.

With thanks to @jhlegarreta for his first contribution in `#1293 <https://github.com/nipreps/mriqc/pull/1293>__`.

.. admonition:: Author list for papers based on *MRIQC* 24.0 series

    As described in the `Contributor Guidelines
    <https://www.nipreps.org/community/CONTRIBUTING/#recognizing-contributions>`__,
    anyone listed as a developer or contributor may write and submit manuscripts
    about *MRIQC*.
    To do so, please move the author(s) name(s) to the front of the following list:

    Christopher J. Markiewicz \ :sup:`1`\ ; Zvi Baratz \ :sup:`2`\ ; Eilidh MacNicol \ :sup:`3`\ ; Céline Provins \ :sup:`4`\ ; Teresa Gomez \ :sup:`5`\ ; Dylan Nielson \ :sup:`6`\ ; Ross W. Blair \ :sup:`1`\ ; Jan Varada \ :sup:`7`\ ; Dimitri Papadopoulos Orfanos \ :sup:`8`\ ; William Triplett \ :sup:`9`\ ; Mathias Goncalves \ :sup:`1`\ ; Nikita Beliy \ :sup:`10`\ ; John A. Lee \ :sup:`11`\ ; Yibei Chen \ :sup:`12`\ ; Ursula A. Tooley \ :sup:`13`\ ; Patrick Sadil \ :sup:`14`\ ; Yaroslav O. Halchenko \ :sup:`15`\ ; James D. Kent \ :sup:`16`\ ; Taylor Salo \ :sup:`17`\ ; Bennet Fauber \ :sup:`18`\ ; Thomas Nichols \ :sup:`19`\ ; Pablo Velasco \ :sup:`20`\ ; Michael Krause \ :sup:`21`\ ; Jon Haitz Legarreta Gorroño \ :sup:`22`\ ; Satrajit S. Ghosh \ :sup:`23`\ ; Joke Durnez \ :sup:`1`\ ; Johannes Achtzehn \ :sup:`24`\ ; Elodie Savary \ :sup:`4`\ ; Adam Huffman \ :sup:`25`\ ; Rafael Garcia-Dias \ :sup:`26`\ ; Michael G. Clark \ :sup:`27`\ ; Michael Dayan \ :sup:`28`\ ; McKenzie P. Hagen \ :sup:`29`\ ; Daniel Birman \ :sup:`1`\ ; Benjamin Kay \ :sup:`30`\ ; Asier Erramuzpe \ :sup:`31`\ ; Adam C. Raikes \ :sup:`32`\ ; Adam G. Thomas \ :sup:`33`\ ; Russell A. Poldrack \ :sup:`1`\ ; Ariel Rokem \ :sup:`5`\ ; Oscar Esteban \ :sup:`4`\ .

    Affiliations:

      1. Department of Psychology, Stanford University, CA, USA
      2. Quantivly Inc., Somerville, MA, USA
      3. Department of Neuroimaging, Institute of Psychiatry, Psychology and Neuroscience, King's College London, London, UK
      4. Department of Radiology, Lausanne University Hospital and University of Lausanne, Switzerland
      5. The University of Washington eScience Institute, WA, USA
      6. Section on Clinical and Computational Psychiatry, National Institute of Mental Health, Bethesda, MD, USA
      7. Functional MRI Facility, National Institute of Mental Health, Bethesda, MD, USA
      8. NeuroSpin, CEA, Université Paris-Saclay, NeuroSpin, Gif-sur-Yvette, France
      9. University of Florida: Gainesville, Florida, US
      10. CRC ULiege, Liege, Belgium
      11. Quansight, Dublin, Ireland
      12. McGovern Institute for Brain Research, Massachusetts Institute of Technology, Cambridge, USA
      13. Department of Neuroscience, University of Pennsylvania, PA, USA
      14. Johns Hopkins Bloomberg School of Public Health, MD, USA
      15. Psychological and Brain Sciences Department, Dartmouth College, NH, USA
      16. Department of Psychology, University of Texas at Austin, TX, USA
      17. Department of Psychology, Florida International University, FL, USA
      18. University of Michigan, Ann Arbor, USA
      19. Oxford Big Data Institute, University of Oxford, Oxford, GB
      20. Center for Brain Imaging, New York University, NY, USA
      21. Max Planck Institute for Human Development, Berlin, Germany
      22. Brigham and Women's Hospital, Mass General Brigham, Harvard Medical School, MA, USA
      23. McGovern Institute for Brain Research, MIT, MA, USA; and Department of Otolaryngology, Harvard Medical School, MA, USA
      24. Charité Berlin, Berlin, Germany
      25. Department of Physics, Imperial College London, London, UK
      26. Institute of Psychiatry, Psychology & Neuroscience, King's College London, London, UK
      27. National Institutes of Health, USA
      28. International Committee of the Red Cross - ICRC, Geneva, Switzerland
      29. Psychology Department, University of Washington, Seattle, WA, USA
      30. Washington University School of Medicine, St.Louis, MO, USA
      31. Computational Neuroimaging Lab, BioCruces Health Research Institute
      32. Center for Innovation in Brain Science, University of Arizona, Tucson, AZ, USA
      33. Data Science and Sharing Team, National Institute of Mental Health, Bethesda, MD, USA

CHANGES
-------

* FIX: Multiecho fMRI crashing with 'unhashable type' errors by @oesteban in https://github.com/nipreps/mriqc/pull/1295
* FIX: Set ``n_procs`` instead of ``num_threads`` on node ``apply_hmc`` by @oesteban in https://github.com/nipreps/mriqc/pull/1309
* FIX: Address memory issues by limiting ``BigPlot``'s parallelization. by @oesteban in https://github.com/nipreps/mriqc/pull/1320
* FIX: Address memory issues in the DWI pipeline by @oesteban in https://github.com/nipreps/mriqc/pull/1323
* FIX: Limit IQMs' node number of processes and, therefore, memory by @oesteban in https://github.com/nipreps/mriqc/pull/1325
* FIX: Resolve numeric overflow in drift estimation node by @oesteban in https://github.com/nipreps/mriqc/pull/1324
* FIX: Revise bugfix #1324 by @oesteban in https://github.com/nipreps/mriqc/pull/1327
* FIX: Remove unreachable code within DWI pipeline by @oesteban in https://github.com/nipreps/mriqc/pull/1328
* ENH: Allow moving the cache folder with an environment variable by @oesteban in https://github.com/nipreps/mriqc/pull/1285
* ENH: Flatten multi-echo lists in circumstances that they fail by @oesteban in https://github.com/nipreps/mriqc/pull/1286
* ENH: Added type hints to config module by @zvi-quantivly in https://github.com/nipreps/mriqc/pull/1288
* ENH: Add test for the CLI parser by @jhlegarreta in https://github.com/nipreps/mriqc/pull/1293
* ENH: Add CLI entry point test by @jhlegarreta in https://github.com/nipreps/mriqc/pull/1294
* ENH: Add a development Dockerfile for testing local changes to the repo. by @rwblair in https://github.com/nipreps/mriqc/pull/1299
* ENH: Crawl dataset's metadata only once and before Nipype's workflow by @oesteban in https://github.com/nipreps/mriqc/pull/1317
* ENH(dMRI): Deal gracefully with small CC masks by @oesteban in https://github.com/nipreps/mriqc/pull/1311
* ENH: Leverage new spun-off apply interface by @oesteban in https://github.com/nipreps/mriqc/pull/1313
* MAINT: Removed personal information from maintainers and updated in contributors by @zvi-quantivly in https://github.com/nipreps/mriqc/pull/1289
* MAINT: Add JHLegarreta to the contributors list by @jhlegarreta in https://github.com/nipreps/mriqc/pull/1301
* MAINT: Flexibilize pandas pinned version by @oesteban in https://github.com/nipreps/mriqc/pull/1310
* MAINT: Remove *Pandas*'s ``FutureWarning`` by @oesteban in https://github.com/nipreps/mriqc/pull/1326
* DOC: Add description of ``summary_fg`` to the documentation by @celprov in https://github.com/nipreps/mriqc/pull/1306
* STY: Apply ruff/flake8-implicit-str-concat rule ISC001 by @DimitriPapadopoulos in https://github.com/nipreps/mriqc/pull/1296
* STY: Format *Jupyter notebooks* by @oesteban in https://github.com/nipreps/mriqc/pull/1321

**Full Changelog**: https://github.com/nipreps/mriqc/compare/24.0.0...24.0.1

24.0.0 (April 17, 2024)
=======================
Initial major release of 2024, featuring the **extraction of IQMs from DWI data**
for the first time in *MRIQC*'s timeline.

CHANGES
-------

* FIX: Bug in *toml* loader crashing with mixed arrays in config (#1281)
* FIX: Remove *DataLad* as a node (#1278)
* FIX: Calculation of trivial shells (#1276)
* FIX: Finalized naming and connection of DWI IQMs (#1272)
* FIX: Enable group reports for DWI (#1266)
* FIX: Address issues that had broken the group reports (#1262)
* FIX: Select filters if modalities are selected (#1261)
* FIX: Make sure new logs and config file output are compatible with parallel processes (#1259)
* FIX: Skip short BOLD runs that break outlier detection (#1120)
* FIX: Revise config save/load and update inputs after dropping (#1245)
* FIX: Drift should not be estimated when less than three low-b volumes present (#1242, #1243)
* FIX: Handle ``NUMEXPR_MAX_THREADS`` like ``OMP_NUM_THREADS`` (#1241)
* FIX: Exclude DWI runs with insufficient orientations or missing bvals (#1240)
* FIX: Avert costly ``BIDSLayout.__repr__`` calls when saving config (#1239)
* FIX: Duplicate node in anatomical workflow (#1234)
* FIX: Typo in ``sorted(..., reverse=True)`` call (#1211)
* ENH: Fail for non-formatted code (#1274)
* ENH: Annotate nodes with ``n_procs`` to allow safe parallelization (#1277)
* ENH: Mechanism to protect config's fields and write out config (#1258)
* ENH: Improve documentation and logging of *SynthStrip*'s model (#1254)
* ENH: Improve logging of runtime (#1253)
* ENH: Expose a command-line option for minimum DWI volumes (#1249)
* ENH: Improve error handling and logging (#1238)
* ENH: Add *b*-vector angular deviations as IQMs (#1233)
* ENH: Move from DTI to DKI with multishell data (#1230)
* ENH: Noise floor estimated with PCA (``dwidenoise``) as an IQM (#1229)
* ENH: Integrate PIESNO noise mask and sigma estimation (#1227)
* ENH: Use MAD for robust estimation of sigma in the CC mask (#1228)
* ENH: Add new IQM for DWI → NDC (#1226)
* ENH: Add FA-based IQMs (nans percentage and degenerate percentage) (#1225)
* ENH: Add computation of spiking voxels mask and percent IQMs (#1224)
* ENH: Adds diffusion-related IQMs. (#1131)
* ENH: Revise summary stats extraction and include controlled roundings (#1219)
* DOC: Add changelog to documentation (#1217)
* MAINT: Added ruff to development dependencies (#1271)
* MAINT: Removed pre-commit from development dependencies (#1269)
* MAINT: Clean up more ``FutureWarning`` issued by *Pandas* (#1257)
* MAINT: Prevent pandas-originating deprecation warning (#1251)
* MAINT: Move GitHub Actions and config files from *flake8* → *ruff* (#1212)
* MAINT: Update contributor affiliation in ``CONTRIBUTORS.md`` (#1214)
* STY: Reformat diffusion workflows module (#1279)
* STY: Applied ruff formatting (#1273)

23.1.1 (March 20, 2024)
=======================
A long-overdue hotfix release addressing many bugs generated during the development
of the new dMRI workflows, and some relating to improvements of the handling of
multi-echo fMRI.
The release also include one year-worth of maintenance actions and a general code
cleanup with *Ruff*.

CHANGES
-------

* FIX: Missing connection to head-motion estimation node in DWI workflow by `@oesteban <https://github.com/@oesteban>`__ in `#1207 <https://github.com/nipreps/mriqc/pull/1207>`__
* FIX: Revise porting to ``Loader`` by `@oesteban <https://github.com/@oesteban>`__ in `#1201 <https://github.com/nipreps/mriqc/pull/1201>`__
* FIX: Revise the last two sloppy merges by `@oesteban <https://github.com/@oesteban>`__ in `#1200 <https://github.com/nipreps/mriqc/pull/1200>`__
* FIX: Move from ``pkg_resources`` to ``niworkflows.data.Loader`` by `@oesteban <https://github.com/@oesteban>`__ in `#1199 <https://github.com/nipreps/mriqc/pull/1199>`__
* FIX: DIPY not listed as a dependency by `@oesteban <https://github.com/@oesteban>`__ in `#1197 <https://github.com/nipreps/mriqc/pull/1197>`__
* FIX: Include ``dwidenoise`` within Docker image by `@oesteban <https://github.com/@oesteban>`__ in `#1196 <https://github.com/nipreps/mriqc/pull/1196>`__
* FIX: Copy name attribute of ``dataset_description.json`` from input dataset by `@celprov <https://github.com/@celprov>`__ in `#1187 <https://github.com/nipreps/mriqc/pull/1187>`__
* FIX: Remove FD as an ``iterfield`` in ``MapNode`` causing crash with ME-BOLD by `@celprov <https://github.com/@celprov>`__ in `#1179 <https://github.com/nipreps/mriqc/pull/1179>`__
* FIX: Incorrect plugin metadata passed to *Report Assembler* by `@oesteban <https://github.com/@oesteban>`__ in `#1188 <https://github.com/nipreps/mriqc/pull/1188>`__
* FIX: Temporary fix of the missing ``"dwi"`` key by `@celprov <https://github.com/@celprov>`__ in `#1174 <https://github.com/nipreps/mriqc/pull/1174>`__
* FIX: Rearrange multi-echo report by `@celprov <https://github.com/@celprov>`__ in `#1164 <https://github.com/nipreps/mriqc/pull/1164>`__
* FIX: Typo in ``inputnode`` field in dMRI masking workflow by `@celprov <https://github.com/@celprov>`__ in `#1165 <https://github.com/nipreps/mriqc/pull/1165>`__
* FIX: Bug in group level workflow by `@celprov <https://github.com/@celprov>`__ in `#1148 <https://github.com/nipreps/mriqc/pull/1148>`__
* FIX: Bugs in DWI workflow by `@celprov <https://github.com/@celprov>`__ in `#1147 <https://github.com/nipreps/mriqc/pull/1147>`__
* FIX: Use simpler DWI reference workflow by `@yibeichan <https://github.com/@yibeichan>`__ in `#1145 <https://github.com/nipreps/mriqc/pull/1145>`__
* FIX: Drop deprecated *Networkx*'s API by `@celprov <https://github.com/@celprov>`__ in `#1137 <https://github.com/nipreps/mriqc/pull/1137>`__
* FIX: Replace ``np.float`` by ``np.float64`` by `@celprov <https://github.com/@celprov>`__ in `#1140 <https://github.com/nipreps/mriqc/pull/1140>`__
* ENH: Improved logging and optimize early checkpoint on subjects by `@oesteban <https://github.com/@oesteban>`__ in `#1198 <https://github.com/nipreps/mriqc/pull/1198>`__
* ENH: Store confound timeseries data by `@psadil <https://github.com/@psadil>`__ in `#1166 <https://github.com/nipreps/mriqc/pull/1166>`__
* ENH: Large overhaul of the functional workflow w/focus on ME-EPI by `@oesteban <https://github.com/@oesteban>`__ in `#1155 <https://github.com/nipreps/mriqc/pull/1155>`__
* ENH: Implement BIDS filters file and drop legacy BIDS querying by `@oesteban <https://github.com/@oesteban>`__ in `#1154 <https://github.com/nipreps/mriqc/pull/1154>`__
* ENH: Swap background and zoomed-in visualizations in anatomical reports by `@oesteban <https://github.com/@oesteban>`__ in `#1151 <https://github.com/nipreps/mriqc/pull/1151>`__
* MAINT: Test on *Python* 3.12 by `@DimitriPapadopoulos <https://github.com/@DimitriPapadopoulos>`__ in `#1156 <https://github.com/nipreps/mriqc/pull/1156>`__
* MAINT: Disable flaky T1w test on CircleCI by `@oesteban <https://github.com/@oesteban>`__ in `#1202 <https://github.com/nipreps/mriqc/pull/1202>`__
* MAINT: Overhaul of the ``Dockerfile`` by `@oesteban <https://github.com/@oesteban>`__ in `#1195 <https://github.com/nipreps/mriqc/pull/1195>`__
* MAINT: Revise package's extra dependencies by `@oesteban <https://github.com/@oesteban>`__ in `#1194 <https://github.com/nipreps/mriqc/pull/1194>`__
* MAINT: Clean up some ``setuptools_scm`` remnants by `@oesteban <https://github.com/@oesteban>`__ in `#1193 <https://github.com/nipreps/mriqc/pull/1193>`__
* MAINT: Load ``FMRISummary`` from *NiReports* rather than *NiWorkflows* by `@celprov <https://github.com/@celprov>`__ in `#1167 <https://github.com/nipreps/mriqc/pull/1167>`__
* MAINT: Update to latest *migas*' API by `@mgxd <https://github.com/@mgxd>`__ in `#1160 <https://github.com/nipreps/mriqc/pull/1160>`__
* MAINT: Update bold to large resource class in ``config.yml`` by `@oesteban <https://github.com/@oesteban>`__ in `#1158 <https://github.com/nipreps/mriqc/pull/1158>`__
* MAINT: Refresh cached intermediate results by `@oesteban <https://github.com/@oesteban>`__ in `#1143 <https://github.com/nipreps/mriqc/pull/1143>`__
* MAINT: Simplify GitHub actions checks and update action versions by `@effigies <https://github.com/@effigies>`__ in `#1141 <https://github.com/nipreps/mriqc/pull/1141>`__
* MAINT: Python 3.11 is supported by `@DimitriPapadopoulos <https://github.com/@DimitriPapadopoulos>`__ in `#1123 <https://github.com/nipreps/mriqc/pull/1123>`__
* MAINT: Apply suggestions from pyupgrade by `@DimitriPapadopoulos <https://github.com/@DimitriPapadopoulos>`__ in `#1124 <https://github.com/nipreps/mriqc/pull/1124>`__
* DOC: Update *Sphinx* pinned version to 5 by `@oesteban <https://github.com/@oesteban>`__ in `#1192 <https://github.com/nipreps/mriqc/pull/1192>`__
* DOC: http:// → https:// by `@DimitriPapadopoulos <https://github.com/@DimitriPapadopoulos>`__ in `#1126 <https://github.com/nipreps/mriqc/pull/1126>`__
* DOC: Add info on the *FreeSurfer* requirement for bare install to address #1034 by `@neurorepro <https://github.com/@neurorepro>`__ in `#1130 <https://github.com/nipreps/mriqc/pull/1130>`__
* STY: Add *Ruff* config and fix all warnings and errors by `@oesteban <https://github.com/@oesteban>`__ in `#1203 <https://github.com/nipreps/mriqc/pull/1203>`__
* STY: Remove extraneous parentheses by `@DimitriPapadopoulos <https://github.com/@DimitriPapadopoulos>`__ in `#1186 <https://github.com/nipreps/mriqc/pull/1186>`__
* STY: Apply a few refurb suggestions by `@DimitriPapadopoulos <https://github.com/@DimitriPapadopoulos>`__ in `#1162 <https://github.com/nipreps/mriqc/pull/1162>`__
* STY: Fix typo found by codespell by `@DimitriPapadopoulos <https://github.com/@DimitriPapadopoulos>`__ in `#1161 <https://github.com/nipreps/mriqc/pull/1161>`__

23.1.0 (June 14, 2023)
======================
A new minor release featuring the new individual reports built with the new
*NiReports* VRS (visual reports system). This means *MRIQC* now uses the same
package *fMRIPrep* uses for generating its reports. In addition to that,
this new release also features *Beta* support for diffusion MRI (dMRI). 

CHANGES
-------

* FIX: Better handling of BIDS cached indexation (`#1121 <https://github.com/nipreps/mriqc/pull/1121>`__)
* FIX: Make doctest of ``NumberOfShells`` more reliable (`#1122 <https://github.com/nipreps/mriqc/pull/1122>`__)
* FIX: Add protection for NaNs and INFs when calculating QI2 (`#1112 <https://github.com/nipreps/mriqc/pull/1112>`__)
* FIX: ``PlotMosaic`` expects lists, not tuples (`#1111 <https://github.com/nipreps/mriqc/pull/1111>`__)
* FIX: BIDS database directory handling (`#1110 <https://github.com/nipreps/mriqc/pull/1110>`__)
* FIX: Remove unused dipy import in the functional interfaces (`#1109 <https://github.com/nipreps/mriqc/pull/1109>`__)
* FIX: Refine the head mask after removal of FSL BET (`#1107 <https://github.com/nipreps/mriqc/pull/1107>`__)
* FIX: Inform *SynthStrip* about the desired intraop threads (`#1101 <https://github.com/nipreps/mriqc/pull/1101>`__)
* FIX: Test broken by #1098 (`#1100 <https://github.com/nipreps/mriqc/pull/1100>`__)
* FIX: Separate report bootstrap files (anat vs. func) (`#1098 <https://github.com/nipreps/mriqc/pull/1098>`__)
* FIX: Propagate logging level to subprocesses (`#1030 <https://github.com/nipreps/mriqc/pull/1030>`__)
* ENH: Incorporate new NiReports' DWI heatmaps (`#1119 <https://github.com/nipreps/mriqc/pull/1119>`__)
* ENH: More compact of shell-wise summary statistic maps (avg/std) (`#1116 <https://github.com/nipreps/mriqc/pull/1116>`__)
* ENH: Add a basic DTI fitting into the diffusion workflow (`#1115 <https://github.com/nipreps/mriqc/pull/1115>`__)
* ENH: MRIQC for DWI (`#1113 <https://github.com/nipreps/mriqc/pull/1113>`__)
* ENH: Culminate dropping FSL as a dependency (`#1108 <https://github.com/nipreps/mriqc/pull/1108>`__)
* ENH: Replace FSL FAST with ANTs Atropos for brain tissue segmentation (`#1099 <https://github.com/nipreps/mriqc/pull/1099>`__)
* ENH: Drop FSL MELODIC (without alternative) (`#1106 <https://github.com/nipreps/mriqc/pull/1106>`__)
* ENH: Drop FSL BET to estimate the "outskin" (head) mask (`#1105 <https://github.com/nipreps/mriqc/pull/1105>`__)
* ENH: Drop utilization of "head" mask from template (`#1104 <https://github.com/nipreps/mriqc/pull/1104>`__)
* ENH: Move templates' probsegs into individual at normalization (`#1103 <https://github.com/nipreps/mriqc/pull/1103>`__)
* ENH: Improving the resource monitor -- infer PID from process name (`#1049 <https://github.com/nipreps/mriqc/pull/1049>`__) (`#1049 <https://github.com/nipreps/mriqc/pull/1049>`__)
* ENH: Refactor reports system to use *NiReports* and the general VRS (`#1085 <https://github.com/nipreps/mriqc/pull/1085>`__)
* MAINT: Move codespell configuration to ``pyproject.toml`` (`#1097 <https://github.com/nipreps/mriqc/pull/1097>`__)
* MAINT: Update deprecated ``nibabel.spatialimage.get_data()`` calls (`#1096 <https://github.com/nipreps/mriqc/pull/1096>`__)

.. admonition:: Author list for papers based on *MRIQC* 23.0 series

    As described in the `Contributor Guidelines
    <https://www.nipreps.org/community/CONTRIBUTING/#recognizing-contributions>`__,
    anyone listed as developer or contributor may write and submit manuscripts
    about *MRIQC*.
    To do so, please move the author(s) name(s) to the front of the following list:
    
    Zvi Baratz \ :sup:`1`\ ; Christopher J. Markiewicz \ :sup:`2`\ ; Eilidh MacNicol \ :sup:`3`\ ; Dylan Nielson \ :sup:`4`\ ; Jan Varada \ :sup:`5`\ ; Ross W. Blair \ :sup:`2`\ ; Céline Provins \ :sup:`6`\ ; William Triplett \ :sup:`7`\ ; Mathias Goncalves \ :sup:`2`\ ; Nikita Beliy \ :sup:`8`\ ; John A. Lee \ :sup:`9`\ ; Ursula A. Tooley \ :sup:`10`\ ; James D. Kent \ :sup:`11`\ ; Yaroslav O. Halchenko \ :sup:`12`\ ; Bennet Fauber \ :sup:`13`\ ; Taylor Salo \ :sup:`14`\ ; Michael Krause \ :sup:`15`\ ; Pablo Velasco \ :sup:`16`\ ; Thomas Nichols \ :sup:`17`\ ; Adam Huffman \ :sup:`18`\ ; Elodie Savary \ :sup:`6`\ ; Johannes Achtzehn \ :sup:`19`\ ; Joke Durnez \ :sup:`2`\ ; Satrajit S. Ghosh \ :sup:`20`\ ; Asier Erramuzpe \ :sup:`21`\ ; Benjamin Kay \ :sup:`22`\ ; Daniel Birman \ :sup:`2`\ ; McKenzie P. Hagen \ :sup:`23`\ ; Michael G. Clark \ :sup:`24`\ ; Patrick Sadil \ :sup:`25`\ ; Rafael Garcia-Dias \ :sup:`26`\ ; Adam G. Thomas \ :sup:`27`\ ; Russell A. Poldrack \ :sup:`2`\ ; Ariel Rokem \ :sup:`28`\ ; Oscar Esteban \ :sup:`6`\ .

    Affiliations:

      1. Sagol School of Neuroscience, Tel Aviv University, Tel Aviv, Israel
      2. Department of Psychology, Stanford University, CA, USA
      3. Department of Neuroimaging, Institute of Psychiatry, Psychology and Neuroscience, King's College London, London, UK
      4. Section on Clinical and Computational Psychiatry, National Institute of Mental Health, Bethesda, MD, USA
      5. Functional MRI Facility, National Institute of Mental Health, Bethesda, MD, USA
      6. Department of Radiology, Lausanne University Hospital and University of Lausanne, Switzerland
      7. University of Florida: Gainesville, Florida, US
      8. CRC ULiege, Liege, Belgium
      9. Quansight, Dublin, Ireland
      10. Department of Neuroscience, University of Pennsylvania, PA, USA
      11. Department of Psychology, University of Texas at Austin, TX, USA
      12. Psychological and Brain Sciences Department, Dartmouth College, NH, USA
      13. University of Michigan, Ann Arbor, USA
      14. Department of Psychology, Florida International University, FL, USA
      15. Max Planck Institute for Human Development, Berlin, Germany
      16. Center for Brain Imaging, New York University, NY, USA
      17. Oxford Big Data Institute, University of Oxford, Oxford, GB
      18. Department of Physics, Imperial College London, London, UK
      19. Charité Berlin, Berlin, Germany
      20. McGovern Institute for Brain Research, MIT, MA, USA; and Department of Otolaryngology, Harvard Medical School, MA, USA
      21. Computational Neuroimaging Lab, BioCruces Health Research Institute
      22. Washington University School of Medicine, St.Louis, MO, USA
      23. Psychology Department, University of Washington, Seattle, WA, USA
      24. National Institutes of Health, USA
      25. Johns Hopkins Bloomberg School of Public Health, MD, USA
      26. Institute of Psychiatry, Psychology & Neuroscience, King's College London, London, UK
      27. Data Science and Sharing Team, National Institute of Mental Health, Bethesda, MD, USA
      28. The University of Washington eScience Institute, WA, USA

23.0.1 (March 24, 2023)
=======================
A hotfix release resolving a reggression introduced with the new optimized indexing.

* FIX: Underspecified regex sets ``BIDSLayout`` to ignore data with sessions (`#1094 <https://github.com/nipreps/mriqc/pull/1094>`__)
* FIX: Input data has incompatible dimensionality (plotting ICA) (`#1082 <https://github.com/nipreps/mriqc/pull/1082>`__)
* ENH: Optimize metadata gathering reusing ``BIDSLayout`` db (`#1084 <https://github.com/nipreps/mriqc/pull/1084>`__)
* DOC : update anatomical example report in documentation (`#1088 <https://github.com/nipreps/mriqc/pull/1088>`__)
* MAINT: Drop old ``mriqc_plot`` script (`#1091 <https://github.com/nipreps/mriqc/pull/1091>`__)

23.0.0 (March 10, 2023)
=======================
The new 23.0.x series include several prominent changes.
Visualization has been migrated from *MRIQC* and *niworkflows* over to the new *NiReports* project.
This series include a major bugfix with **the optimization of the indexing** of the input BIDS folder,
which was taking large times with sizeable datasets.
Telemetry has also been incorporated with *migas*.
These new series also involve maintenance housekeeping, and includes some relevant bugfixes.

New contributors
----------------

* `@arokem <https://github.com/arokem>`__ made their first contribution in `#1040 <https://github.com/nipreps/mriqc/pull/1040>`__
* `@yarikoptic <https://github.com/yarikoptic>`__ made their first contribution in `#1057 <https://github.com/nipreps/mriqc/pull/1057>`__
* `@esavary <https://github.com/esavary>`__ made their first contribution in `#1047 <https://github.com/nipreps/mriqc/pull/1047>`__

CHANGES
-------
**Full Changelog**: https://github.com/nipreps/mriqc/compare/22.0.6...23.0.0

* FIX: Send metadata extraction to workers (functional workflow) (`#1081 <https://github.com/nipreps/mriqc/pull/1081>`__)
* FIX: Plot coronal as main plain for mosaic of rodent images (`#1027 <https://github.com/nipreps/mriqc/pull/1027>`__)
* FIX: Address non-empty take from empty axes (anatomical IQMs) (`#1077 <https://github.com/nipreps/mriqc/pull/1077>`__)
* FIX: Uniformize building workflow message (anat vs. func) (`#1072 <https://github.com/nipreps/mriqc/pull/1072>`__)
* FIX: Move telemetry atexit into entrypoint func (`#1067 <https://github.com/nipreps/mriqc/pull/1067>`__)
* FIX: Preempt PyBIDS to spend time indexing non-BIDS folders (`#1050 <https://github.com/nipreps/mriqc/pull/1050>`__)
* FIX: Update T1w metrics (`#1063 <https://github.com/nipreps/mriqc/pull/1063>`__)
* FIX: Resource monitor would not ever start tracking (`#1051 <https://github.com/nipreps/mriqc/pull/1051>`__)
* ENH: Add DataLad getter to inputs of functional workflows (`#1071 <https://github.com/nipreps/mriqc/pull/1071>`__)
* ENH: Add migas telemetry (`#1036 <https://github.com/nipreps/mriqc/pull/1036>`__)
* ENH: Add codespell automation: config, action, and typos fixed (`#1057 <https://github.com/nipreps/mriqc/pull/1057>`__)
* MAINT: Update *NiReports* calls to upcoming interfaces API (`#1078 <https://github.com/nipreps/mriqc/pull/1078>`__)
* MAINT: Pacify codespell (`#1080 <https://github.com/nipreps/mriqc/pull/1080>`__)
* MAINT: Conclude porting of reportlets into *NiReports* (`#1068 <https://github.com/nipreps/mriqc/pull/1068>`__)
* MAINT: Migrate to hatchling (`#1070 <https://github.com/nipreps/mriqc/pull/1070>`__)
* MAINT: Pin PyBIDS 0.15.6 (culminating #1050) (`#1069 <https://github.com/nipreps/mriqc/pull/1069>`__)
* MAINT: Update niworkflows pin to support newer ANTs releases (`#1047 <https://github.com/nipreps/mriqc/pull/1047>`__)
* MAINT: Fix minor aspects of WebAPI deployment on CircleCI (`#1064 <https://github.com/nipreps/mriqc/pull/1064>`__)
* MAINT: Update CircleCI executor and use built-in docker-compose (`#1061 <https://github.com/nipreps/mriqc/pull/1061>`__)
* MAINT: Rotate CircleCI secrets and setup up org-level context (`#1046 <https://github.com/nipreps/mriqc/pull/1046>`__)
* DOC: Update documentation with the new carpet plot (`#1045 <https://github.com/nipreps/mriqc/pull/1045>`__)
* DOC: Complete the documentation of ``summary_stats()`` (`#1044 <https://github.com/nipreps/mriqc/pull/1044>`__)
* DOC: Fixes a couple of broken links to the *nipype* documentation (`#1040 <https://github.com/nipreps/mriqc/pull/1040>`__)

.. admonition:: Author list for papers based on *MRIQC* 23.0 series

    As described in the `Contributor Guidelines
    <https://www.nipreps.org/community/CONTRIBUTING/#recognizing-contributions>`__,
    anyone listed as developer or contributor may write and submit manuscripts
    about *MRIQC*.
    To do so, please move the author(s) name(s) to the front of the following list:

    Zvi Baratz \ :sup:`1`\ ; Christopher J. Markiewicz \ :sup:`2`\ ; Eilidh MacNicol \ :sup:`3`\ ; Dylan Nielson \ :sup:`4`\ ; Jan Varada \ :sup:`5`\ ; Ross W. Blair \ :sup:`2`\ ; Céline Provins \ :sup:`6`\ ; William Triplett \ :sup:`7`\ ; Mathias Goncalves \ :sup:`2`\ ; Nikita Beliy \ :sup:`8`\ ; John A. Lee \ :sup:`9`\ ; Ursula A. Tooley \ :sup:`10`\ ; James D. Kent \ :sup:`11`\ ; Yaroslav O. Halchenko \ :sup:`12`\ ; Bennet Fauber \ :sup:`13`\ ; Taylor Salo \ :sup:`14`\ ; Michael Krause \ :sup:`15`\ ; Pablo Velasco \ :sup:`16`\ ; Thomas Nichols \ :sup:`17`\ ; Adam Huffman \ :sup:`18`\ ; Johannes Achtzehn \ :sup:`19`\ ; Joke Durnez \ :sup:`2`\ ; Satrajit S. Ghosh \ :sup:`20`\ ; Asier Erramuzpe \ :sup:`21`\ ; Benjamin Kay \ :sup:`22`\ ; Daniel Birman \ :sup:`2`\ ; Elodie Savary \ :sup:`23`\ ; McKenzie P. Hagen \ :sup:`24`\ ; Michael G. Clark \ :sup:`25`\ ; Patrick Sadil \ :sup:`26`\ ; Rafael Garcia-Dias \ :sup:`27`\ ; Adam G. Thomas \ :sup:`28`\ ; Russell A. Poldrack \ :sup:`2`\ ; Ariel Rokem \ :sup:`29`\ ; Oscar Esteban \ :sup:`30`\ .

    Affiliations:

      1. Sagol School of Neuroscience, Tel Aviv University, Tel Aviv, Israel
      2. Department of Psychology, Stanford University, CA, USA
      3. Department of Neuroimaging, Institute of Psychiatry, Psychology and Neuroscience, King's College London, London, UK
      4. Section on Clinical and Computational Psychiatry, National Institute of Mental Health, Bethesda, MD, USA
      5. Functional MRI Facility, National Institute of Mental Health, Bethesda, MD, USA
      6. Lausanne University Hospital and University of Lausanne, Lausanne, Switzerland
      7. University of Florida: Gainesville, Florida, US
      8. CRC ULiege, Liege, Belgium
      9. Quansight, Dublin, Ireland
      10. Department of Neuroscience, University of Pennsylvania, PA, USA
      11. Department of Psychology, University of Texas at Austin, TX, USA
      12. Psychological and Brain Sciences Department, Dartmouth College, NH, USA
      13. University of Michigan, Ann Arbor, USA
      14. Department of Psychology, Florida International University, FL, USA
      15. Max Planck Institute for Human Development, Berlin, Germany
      16. Center for Brain Imaging, New York University, NY, USA
      17. Oxford Big Data Institute, University of Oxford, Oxford, GB
      18. Department of Physics, Imperial College London, London, UK
      19. Charité Berlin, Berlin, Germany
      20. McGovern Institute for Brain Research, MIT, MA, USA; and Department of Otolaryngology, Harvard Medical School, MA, USA
      21. Computational Neuroimaging Lab, BioCruces Health Research Institute
      22. Washington University School of Medicine, St.Louis, MO, USA
      23. Department of Radiology, Lausanne University Hospital and University of Lausanne, Switzerland
      24. Psychology Department, University of Washington, Seattle, WA, USA
      25. National Institutes of Health, USA
      26. Johns Hopkins Bloomberg School of Public Health, MD, USA
      27. Institute of Psychiatry, Psychology & Neuroscience, King's College London, London, UK
      28. Data Science and Sharing Team, National Institute of Mental Health, Bethesda, MD, USA
      29. The University of Washington eScience Institute, WA, USA
      30. Department of Radiology, Lausanne University Hospital and University of Lausanne

22.0.6 (August 24, 2022)
========================
A hotfix release partially rolling-back the previous fix #1025.
Thanks everyone for your patience with the excessively rushed release of 22.0.5.

* FIX: Better fix to the multi-argument ``--participant-label`` issue (`#1026 <https://github.com/nipreps/mriqc/pull/1026>`__)

22.0.5 (August 24, 2022)
========================
A hotfix release addressing a problem with the argument parser.

* FIX: Multiple valued ``--participant-label`` wrongly parsed (`#1025 <https://github.com/nipreps/mriqc/pull/1025>`__)

22.0.4 (August 23, 2022)
========================
A hotfix release to ensure smooth operation of datalad within Docker.

* FIX: Major improvements to new datalad-based interface & perform within containers (`#1024 <https://github.com/nipreps/mriqc/pull/1024>`__)
* ENH: Bump Docker base to latest release (`#1022 <https://github.com/nipreps/mriqc/pull/1022>`__)

22.0.3 (August 19, 2022)
========================
A patch release containing a bugfix to the SynthStrip preprocessing.

* FIX: SynthStrip preprocessing miscalculating new shape after reorientation (`#1021 <https://github.com/nipreps/mriqc/pull/1021>`__)
* ENH: Remove slice-timing correction (`#1019 <https://github.com/nipreps/mriqc/pull/1019>`__)
* ENH: Add a new ``DataladIdentityInterface`` (`#1020 <https://github.com/nipreps/mriqc/pull/1020>`__)
* ENH: Set rat-specific defaults for FD calculations (`#1005 <https://github.com/nipreps/mriqc/pull/1005>`__)
* ENH: New version of the rating widget (`#1012 <https://github.com/nipreps/mriqc/pull/1012>`__)
* DOC: Move readthedocs to use the config v2 file (YAML) (`#1018 <https://github.com/nipreps/mriqc/pull/1018>`__)
* MAINT: Fix statsmodels dependency, it is not optional (`#1017 <https://github.com/nipreps/mriqc/pull/1017>`__)
* MAINT: Several critical updates to CircleCI and Docker images (`#1016 <https://github.com/nipreps/mriqc/pull/1016>`__)
* MAINT: Update the T1w IQMs to the new reference after #997 (`#1014 <https://github.com/nipreps/mriqc/pull/1014>`__)
* MAINT: Fix failing tests as ``python setup.py`` is deprecated (`#1013 <https://github.com/nipreps/mriqc/pull/1013>`__)

22.0.2 (August 15, 2022)
========================
A patch release including the new ratings widget.

* ENH: New version of the rating widget (`#1012 <https://github.com/nipreps/mriqc/pull/1012>`__)
* DOC: Move readthedocs to use the config v2 file (YAML) (`#1018 <https://github.com/nipreps/mriqc/pull/1018>`__)
* MAINT: Fix ``statsmodels`` dependency, it is not optional (`#1017 <https://github.com/nipreps/mriqc/pull/1017>`__)
* MAINT: Several critical updates to CircleCI and Docker images (`#1016 <https://github.com/nipreps/mriqc/pull/1016>`__)
* MAINT: Update the T1w IQMs to the new reference after #997 (`#1014 <https://github.com/nipreps/mriqc/pull/1014>`__)
* MAINT: Fix failing tests as ``python setup.py`` is deprecated (`#1013 <https://github.com/nipreps/mriqc/pull/1013>`__)

22.0.1 (May 3rd, 2022)
======================
A patch release addressing a new minor bug.

* FIX: More lenient handling of skull-stripped datasets (`#997 <https://github.com/nipreps/mriqc/pull/997>`__)

22.0.0 (May 3rd, 2022)
======================
First official release after migrating the repository into the *NiPreps*' organization.
A major new feature is the rodent pipeline by Eilidh MacNicol (@eilidhmacnicol).
A second major feature is the adoption of the updated carpet plots for BOLD fMRI,
contributed by Céline Provins (@celprov).
Virtual memory allocation has been ten-fold cut down, and a complementary resource monitor instrumentation is now available with *MRIQC*.
This release updates the Docker image with up-to-date dependencies, updates
*MRIQC*'s codebase to the latest *NiTransforms* and includes some minor bugfixes.
The code, modules and data related to the MRIQC classifier have been extracted into an
isolated package called [*MRIQC-learn*](https://github.com/nipreps/mriqc-learn).
Finally, this release also contains a major code style overhaul by Zvi Baratz.

The contributor/author crediting system has been adapted to the current draft of the
*NiPreps Community* Governance documents.

With thanks to @ZviBaratz, @nbeliy, @octomike, @benkay86, @verdurin, @leej3, @utooley,
and @jAchtzehn for their contributions.

* FIX: Inconsistent API in anatomical CNR computation (`#995 <https://github.com/nipreps/mriqc/pull/995>`__)
* FIX: Check sanity of input data before extracting IQMs (`#994 <https://github.com/nipreps/mriqc/pull/994>`__)
* FIX: Plot segmentations after dropping off-diagonal (`#989 <https://github.com/nipreps/mriqc/pull/989>`__)
* FIX: Replace all deprecated ``nibabel.get_data()`` in anatomical module (`#988 <https://github.com/nipreps/mriqc/pull/988>`__)
* FIX: Resource profiler was broken with config file (`#981 <https://github.com/nipreps/mriqc/pull/981>`__)
* FIX: preserve WM segments in rodents (`#979 <https://github.com/nipreps/mriqc/pull/979>`__)
* FIX: Pin ``jinja2 < 3.1`` (`#978 <https://github.com/nipreps/mriqc/pull/978>`__)
* FIX: Make toml config unique, works around #912 (`#960 <https://github.com/nipreps/mriqc/pull/960>`__)
* FIX: Nipype multiproc plugin expects ``n_procs`` and not ``nprocs`` (`#961 <https://github.com/nipreps/mriqc/pull/961>`__)
* FIX: Set TR when generating carpetplots (enables time for X axis) (`#971 <https://github.com/nipreps/mriqc/pull/971>`__)
* FIX: ``template_resolution`` deprecation warning (`#941 <https://github.com/nipreps/mriqc/pull/941>`__)
* FIX: Set entity ``datatype`` in ``BIDSLayout`` queries (`#942 <https://github.com/nipreps/mriqc/pull/942>`__)
* FIX: T2w image of MNI template unavailable in Singularity (`#940 <https://github.com/nipreps/mriqc/pull/940>`__)
* FIX: Release process -- Docker deployment not working + Python package lacks WebAPI token (`#938 <https://github.com/nipreps/mriqc/pull/938>`__)
* FIX: Revise building documentation at RTD after migration (`#935 <https://github.com/nipreps/mriqc/pull/935>`__)
* FIX: Final touch-ups in the maintenance of Docker image + CI (`#928 <https://github.com/nipreps/mriqc/pull/928>`__)
* FIX: Update unit tests (`#927 <https://github.com/nipreps/mriqc/pull/927>`__)
* FIX: Update dependencies and repair BOLD workflow accordingly (`#926 <https://github.com/nipreps/mriqc/pull/926>`__)
* FIX: Update dependencies and repair T1w workflow accordingly (`#925 <https://github.com/nipreps/mriqc/pull/925>`__)
* FIX: Set ``matplotlib`` on ``Agg`` output mode (`#892 <https://github.com/nipreps/mriqc/pull/892>`__)
* ENH: Deprecate ``--start-idx`` / ``--stop-idx`` (`#993 <https://github.com/nipreps/mriqc/pull/993>`__)
* ENH: Add SynthStrip base module (`#987 <https://github.com/nipreps/mriqc/pull/987>`__)
* ENH: Improve building workflow message feedback (`#990 <https://github.com/nipreps/mriqc/pull/990>`__)
* ENH: Add instrumentation to monitor resources (`#984 <https://github.com/nipreps/mriqc/pull/984>`__)
* ENH: Standalone, lightweight version of MultiProc plugin (`#985 <https://github.com/nipreps/mriqc/pull/985>`__)
* ENH: Revise plugin and workflow initialization (`#983 <https://github.com/nipreps/mriqc/pull/983>`__)
* ENH: Base generalization of the pipeline for rodents (`#969 <https://github.com/nipreps/mriqc/pull/969>`__)
* ENH: Update to new *NiWorkflows*' API, which adds the crown to the carpetplot (`#968 <https://github.com/nipreps/mriqc/pull/968>`__)
* ENH: Optimize *PyBIDS*' layout initialization (`#939 <https://github.com/nipreps/mriqc/pull/939>`__)
* ENH: Refactored long strings to a :mod:`mriqc.messages` module (`#901 <https://github.com/nipreps/mriqc/pull/901>`__)
* ENH: Refactored :mod:`mriqc.interfaces.common` module (`#901 <https://github.com/nipreps/mriqc/pull/901>`__)
* DOC: Improve documentation of ``--nprocs`` and ``--omp-nthreads`` (`#986 <https://github.com/nipreps/mriqc/pull/986>`__)
* DOC: Add ``sbatch`` file example for SLURM execution (`#963 <https://github.com/nipreps/mriqc/pull/963>`__)
* DOC: Various fixes to "Running mriqc" section (`#897 <https://github.com/nipreps/mriqc/pull/897>`__)
* MAINT: Refactor ``Dockerfile`` using new miniconda image (`#974 <https://github.com/nipreps/mriqc/pull/974>`__)
* MAINT: Outsource the classifier into nipreps/mriqc-learn (`#973 <https://github.com/nipreps/mriqc/pull/973>`__)
* MAINT: Update ``CONTRIBUTORS.md`` (`#953 <https://github.com/nipreps/mriqc/pull/953>`__)
* MAINT: Update contributor location (`#952 <https://github.com/nipreps/mriqc/pull/952>`__)
* MAINT: Updates to ``CONTRIBUTORS.md`` file
* MAINT: Revise Docker image settings & CircleCI (`#937 <https://github.com/nipreps/mriqc/pull/937>`__)
* MAINT: Finalize transfer to ``nipreps`` organization (`#936 <https://github.com/nipreps/mriqc/pull/936>`__)
* MAINT: Relicensing to Apache-2.0, for compliance with *NiPreps* and prior transfer to the org (`#930 <https://github.com/nipreps/mriqc/pull/930>`__)
* MAINT: New Docker layer caching system of other *NiPreps* (`#929 <https://github.com/nipreps/mriqc/pull/929>`__)
* MAINT: Code style overhaul (`#901 <https://github.com/nipreps/mriqc/pull/901>`__)
* MAINT: Update ``Dockerfile`` and catch-up with *fMRIPrep*'s (`#924 <https://github.com/nipreps/mriqc/pull/924>`__)
* STY: Run ``black`` at the top of the repo (`#932 <https://github.com/nipreps/mriqc/pull/932>`__)

**Full Changelog**: https://github.com/nipreps/mriqc/compare/0.16.1...22.0.0

.. admonition:: Author list for papers based on *MRIQC* 22.0.x

    As described in the `Contributor Guidelines
    <https://www.nipreps.org/community/CONTRIBUTING/#recognizing-contributions>`__,
    anyone listed as developer or contributor may write and submit manuscripts
    about *MRIQC*.
    To do so, please move the author(s) name(s) to the front of the following list:

    Zvi Baratz \ :sup:`1`\ ; Christopher J. Markiewicz \ :sup:`2`\ ; Eilidh MacNicol \ :sup:`3`\ ; Dylan Nielson \ :sup:`4`\ ; Jan Varada \ :sup:`5`\ ; Ross W. Blair \ :sup:`2`\ ; William Triplett \ :sup:`6`\ ; Nikita Beliy \ :sup:`7`\ ; Céline Provins \ :sup:`8`\ ; John A. Lee \ :sup:`9`\ ; Ursula A. Tooley \ :sup:`10`\ ; James D. Kent \ :sup:`11`\ ; Bennet Fauber \ :sup:`12`\ ; Taylor Salo \ :sup:`13`\ ; Mathias Goncalves \ :sup:`2`\ ; Michael Krause \ :sup:`14`\ ; Pablo Velasco \ :sup:`15`\ ; Thomas Nichols \ :sup:`16`\ ; Adam Huffman \ :sup:`17`\ ; Johannes Achtzehn \ :sup:`18`\ ; Joke Durnez \ :sup:`2`\ ; Satrajit S. Ghosh \ :sup:`19`\ ; Asier Erramuzpe \ :sup:`20`\ ; Benjamin Kay \ :sup:`21`\ ; Daniel Birman \ :sup:`2`\ ; Michael G. Clark \ :sup:`22`\ ; Rafael Garcia-Dias \ :sup:`23`\ ; Sean Marret \ :sup:`5`\ ; Adam G. Thomas \ :sup:`24`\ ; Russell A. Poldrack \ :sup:`2`\ ; Krzysztof J. Gorgolewski \ :sup:`25`\ ; Oscar Esteban \ :sup:`26`\ .

    Affiliations:

    1. Sagol School of Neuroscience, Tel-Aviv University
    2. Department of Psychology, Stanford University, CA, USA
    3. Department of Neuroimaging, Institute of Psychiatry, Psychology and Neuroscience, King's College London, London, UK
    4. Section on Clinical and Computational Psychiatry, National Institute of Mental Health, Bethesda, MD, USA
    5. Functional MRI Facility, National Institute of Mental Health, Bethesda, MD, USA
    6. University of Florida: Gainesville, Florida, US
    7. CRC ULiege, Liege, Belgium
    8. Lausanne University Hospital and University of Lausanne, Lausanne, Switzerland
    9. Quansight, Dublin, Ireland
    10. Department of Neuroscience, University of Pennsylvania, PA, USA
    11. Department of Psychology, University of Texas at Austin, TX, USA
    12. University of Michigan, Ann Arbor, USA
    13. Department of Psychology, Florida International University, FL, USA
    14. Max Planck Institute for Human Development, Berlin, Germany
    15. Center for Brain Imaging, New York University, NY, USA
    16. Oxford Big Data Institute, University of Oxford, Oxford, GB
    17. Department of Physics, Imperial College London, London, UK
    18. Charité Berlin, Berlin, Germany
    19. McGovern Institute for Brain Research, MIT, MA, USA; and Department of Otolaryngology, Harvard Medical School, MA, USA
    20. Computational Neuroimaging Lab, BioCruces Health Research Institute
    21. Washington University School of Medicine, St.Louis, MO, USA
    22. National Institutes of Health, USA
    23. Institute of Psychiatry, Psychology & Neuroscience, King's College London, London, UK
    24. Data Science and Sharing Team, National Institute of Mental Health, Bethesda, MD, USA
    25. Google LLC
    26. Department of Radiology, Lausanne University Hospital and University of Lausanne

Series 0.16.x
=============
0.16.1 (January 30, 2021)
-------------------------
Bug-fix release in 0.16.x series.

This PR improves BIDS Derivatives compliance, fixes an issue with reading datasets with
subjects of the form ``sub-sXYZ``, and improves compatibility with more recent matplotlib.

* FIX: Participant labels starting with ``[sub]`` cannot be used (`#890 <https://github.com/nipreps/mriqc/pull/890>`__)
* FIX: Change deprecated ``normed`` to ``density`` in parameters to ``hist()`` (`#888 <https://github.com/nipreps/mriqc/pull/888>`__)
* ENH: Write derivatives metadata (`#885 <https://github.com/nipreps/mriqc/pull/885>`__)
* ENH: Add ``--pdb`` option to make debugging easier (`#884 <https://github.com/nipreps/mriqc/pull/884>`__)

0.16.0 (January 5, 2021)
------------------------
New feature release in 0.16.x series.

This version removes the FSL dependency from the fMRI workflow.

* FIX: Skip version cache on read-only filesystems (`#862 <https://github.com/nipreps/mriqc/pull/862>`__)
* FIX: Honor ``$OMP_NUM_THREADS`` environment variable (`#848 <https://github.com/nipreps/mriqc/pull/848>`__)
* RF: Simplify comprehensions, using easy-to-read var names (`#875 <https://github.com/nipreps/mriqc/pull/875>`__)
* RF: Free the fMRI workflow from FSL (`#842 <https://github.com/nipreps/mriqc/pull/842>`__)
* CI: Fix up Circle builds (`#876 <https://github.com/nipreps/mriqc/pull/876>`__)
* CI: Update machine images on Circle (`#874 <https://github.com/nipreps/mriqc/pull/874>`__)

Older (unsupported) series
==========================
0.15.3 (September 18, 2020)
---------------------------
A bugfix release to re-enable setting of ``--omp-nthreads/--ants-nthreads``.

* FIX: ``omp_nthreads`` typo (`#846 <https://github.com/nipreps/mriqc/pull/846>`__)

0.15.2 (April 6, 2020)
----------------------
A bugfix release containing mostly maintenance actions and documentation
improvements. This version drops Python 3.5.
The core of MRIQC has adopted the config-module pattern from fMRIPrep.
With thanks to A. Erramuzpe, @justbennet, U. Tooley, and A. Huffman
for contributions.

* MAINT: revise style of all files (except for workflows) (`#839 <https://github.com/nipreps/mriqc/pull/839>`__)
* MAINT: Clear the clutter of warnings (`#838 <https://github.com/nipreps/mriqc/pull/838>`__)
* RF: Adopt config module pattern from *fMRIPrep* (`#837 <https://github.com/nipreps/mriqc/pull/837>`__)
* MAINT: Clear the clutter of warnings (`#838 <https://github.com/nipreps/mriqc/pull/838>`__)
* MAINT: Drop Python 3.5, simplify linting (`#833 <https://github.com/nipreps/mriqc/pull/833>`__)
* MAINT: Update to latest Ubuntu Xenial tag (`#814 <https://github.com/nipreps/mriqc/pull/814>`__)
* MAINT: Centralize all requirements and versions on ``setup.cfg`` (`#819 <https://github.com/nipreps/mriqc/pull/819>`__)
* MAINT: Use recent Python image to build packages in CircleCI (`#808 <https://github.com/nipreps/mriqc/pull/808>`__)
* DOC: Improve AQI (and other IQMs) and boxplot whiskers descriptions (`#816 <https://github.com/nipreps/mriqc/pull/816>`__)
* DOC: Refactor how documentation is built on CircleCI (`#818 <https://github.com/nipreps/mriqc/pull/818>`__)
* DOC: Corrected a couple of typos in ``--help`` text (`#809 <https://github.com/nipreps/mriqc/pull/809>`__)

0.15.1 (July 26, 2019)
----------------------
A maintenance patch release updating PyBIDS.

* FIX: ``FileNotFoundError`` when MELODIC (``--ica``) does not converge (`#800 <https://github.com/nipreps/mriqc/pull/800>`__) @oesteban
* MAINT: Migrate MRIQC to a ``setup.cfg`` style of installation (`#799 <https://github.com/nipreps/mriqc/pull/799>`__) @oesteban
* MAINT: Use PyBIDS 0.9.2+ via niworkflows PR (`#796 <https://github.com/nipreps/mriqc/pull/796>`__) @effigies

0.15.0 (April 5, 2019)
----------------------
A long overdue update, pinning updated versions of
`TemplateFlow <https://doi.org/10.5281/zenodo.2583289>`__ and
`Niworkflows <https://github.com/nipreps/niworkflows>`__.
With thanks to @garciadias for contributions.

* ENH: Revision of QI2 (`#606 <https://github.com/nipreps/mriqc/pull/606>`__) @oesteban
* FIX: Set matplotlib backend early (`#759 <https://github.com/nipreps/mriqc/pull/759>`__) @oesteban
* FIX: Niworkflows pin <0.5 (`#766 <https://github.com/nipreps/mriqc/pull/766>`__) @oesteban
* DOC: Update BIDS validation link. (`#764 <https://github.com/nipreps/mriqc/pull/764>`__) @garciadias
* DOC: Add data sharing agreement (`#765 <https://github.com/nipreps/mriqc/pull/765>`__) @oesteban
* FIX: Catch uncaught exception in WebAPI upload. (`#774 <https://github.com/nipreps/mriqc/pull/774>`__) @rwblair
* FIX/DOC: Append new line after dashes in ``mriqc_run`` help text (`#777 <https://github.com/nipreps/mriqc/pull/777>`__) @rwblair
* ENH: Use TemplateFlow and niworkflows-0.8.x (`#782 <https://github.com/nipreps/mriqc/pull/782>`__) @oesteban
* FIX: Correctly set WebAPI rating endpoint in BOLD reports. (`#785 <https://github.com/nipreps/mriqc/pull/785>`__) @oesteban
* FIX: Correctly process values of rating widget (`#787 <https://github.com/nipreps/mriqc/pull/787>`__) @oesteban

0.14.2 (August 20, 2018)
------------------------

* FIX: Preempt pandas resolving ``Path`` objects (`#746 <https://github.com/nipreps/mriqc/pull/746>`__) @oesteban
* FIX: Codacy issues (`#745 <https://github.com/nipreps/mriqc/pull/745>`__) @oesteban

0.14.1 (August 20, 2018)
------------------------

* FIX: Calculate relative path with sessions (`#742 <https://github.com/nipreps/mriqc/pull/742>`__) @oesteban
* ENH: Add a toggle button to rating widget (`#743 <https://github.com/nipreps/mriqc/pull/743>`__) @oesteban

0.14.0 (August 17, 2018)
------------------------

* ENH: New feedback widget (`#740 <https://github.com/nipreps/mriqc/pull/740>`__) @oesteban

0.13.1 (August 16, 2018)
------------------------

* [ENH,FIX] Updates to individual reports, fix table after rating (`#739 <https://github.com/nipreps/mriqc/pull/739>`__) @oesteban

0.13.0 (August 15, 2018)
------------------------

* MAINT: Overdue refactor (`#736 <https://github.com/nipreps/mriqc/pull/736>`__) @oesteban
  * FIX: Reorganize outputs (closes #396)
  * ENH: Memory usage - lessons learned with FMRIPREP (`#703 <https://github.com/nipreps/mriqc/pull/703>`__)
  * FIX: Cannot allocate memory (v 0.9.4) (closes #536)
  * FIX: Drop inoperative ``--report-dir`` flag (`#550 <https://github.com/nipreps/mriqc/pull/550>`__)
  * FIX: Drop misleading WARNING of the group-level execution (`#714 <https://github.com/nipreps/mriqc/pull/714>`__)
  * FIX: Expand usernames on input paths (`#721 <https://github.com/nipreps/mriqc/pull/721>`__)
  * MAINT: More robust naming of derivatives (related to #661)

* FIX: Do not fail with spurious 4th dimension on T1w (`#738 <https://github.com/nipreps/mriqc/pull/738>`__) @oesteban
* ENH: Move on to .tsv files (`#737 <https://github.com/nipreps/mriqc/pull/737>`__) @oesteban

0.12.1 (August 13, 2018)
------------------------

* FIX: ``BIDSLayout`` queries (`#735 <https://github.com/nipreps/mriqc/pull/735>`__)


0.12.0 (August 09, 2018)
------------------------

* FIX: Reduce tSNR memory requirements (`#712 <https://github.com/nipreps/mriqc/pull/712>`__)
* DOC: Fix typos in IQM documentation (`#725 <https://github.com/nipreps/mriqc/pull/725>`__)
* PIN: Update MRIQC WebAPI version (`#734 <https://github.com/nipreps/mriqc/pull/734>`__)
* BUG: Fix missing library in singularity images (`#733 <https://github.com/nipreps/mriqc/pull/733>`__)
* PIN: nipype 1.1.0, niworkflows (`#726 <https://github.com/nipreps/mriqc/pull/726>`__)

0.11.0 (June 05, 2018)
----------------------

* RF: Resume external nipype dependency (`#715 <https://github.com/nipreps/mriqc/pull/715>`__)

0.10.6 (May 29, 2018)
---------------------

* HOTFIX: Bug #659

0.10.5 (May 28, 2018)
---------------------

* ENH: Report feedback (`#659 <https://github.com/nipreps/mriqc/pull/659>`__)

0.10.4 (March 22, 2018)
-----------------------

* ENH: Various improvements to reports (`#708 <https://github.com/nipreps/mriqc/pull/708>`__)
* MAINT: Style revision (`#704 <https://github.com/nipreps/mriqc/pull/704>`__)
* PIN: pybids 0.5 (`#700 <https://github.com/nipreps/mriqc/pull/700>`__)
* ENH: Increase FAST memory limits (`#702 <https://github.com/nipreps/mriqc/pull/702>`__)

0.10.3 (February 26, 2018)
--------------------------

* ENH: Enable T2w metrics uploads (`#696 <https://github.com/nipreps/mriqc/pull/696>`__)
* PIN: Updating niworkflows (`#698 <https://github.com/nipreps/mriqc/pull/698>`__)
* DOC: Option ``-o`` is outdated for classifier (`#697 <https://github.com/nipreps/mriqc/pull/697>`__)

0.10.2 (February 15, 2018)
--------------------------

* ENH: Add warning about mounting relative paths (`#690 <https://github.com/nipreps/mriqc/pull/690>`__)
* FIX: Sanitize inputs (`#687 <https://github.com/nipreps/mriqc/pull/687>`__)
* DOC: Fix documentation to use ``--version`` instead of ``-v`` (`#688 <https://github.com/nipreps/mriqc/pull/688>`__)

0.10.1
------

* FIX: Fixed a bug in reading outputs of ``3dFWHMx`` (`#678 <https://github.com/nipreps/mriqc/pull/678>`__)

0.9.10
------

* FIX: Updated AFNI to 17.3.03. Resolves errors regarding opening display by ``3dSkullStrip`` (`#669 <https://github.com/nipreps/mriqc/pull/669>`__)

0.9.9
-----

* ENH: Update nipype to fix ``$DISPLAY`` problem of AFNI's ``3dSkullStrip``

0.9.8
-----
With thanks to Jan Varada (@jvarada) for the session/run filtering.

* ENH: Report recall in cross-validation (requested by reviewer) (`#633 <https://github.com/nipreps/mriqc/pull/633>`__)
* ENH: Hotfixes to 0.9.7 (`#635 <https://github.com/nipreps/mriqc/pull/635>`__)
* FIX: Implement filters for session, run and task of BIDS input (`#612 <https://github.com/nipreps/mriqc/pull/612>`__)

0.9.7
-----

* ENH: Clip outliers in FD and SPIKES group plots (`#593 <https://github.com/nipreps/mriqc/pull/593>`__)
* ENH: Second revision of the classifier (`#555 <https://github.com/nipreps/mriqc/pull/555>`__):
  * Set matplotlib plugin to `agg` in docker image
  * Migrate scalings to sklearn pipelining system
  * Add Satra's feature selection for RFC (with thanks to S. Ghosh for his suggestion)
  * Make model selection compatible with sklearn `Pipeline`
  * Multiclass classification
  * Add feature selection filter based on Sites prediction (requires pinning to development sklearn-0.19)
  * Add `RobustLeavePGroupsOut`, replace `RobustGridSearchCV` with the standard `GridSearchCV` of sklearn.
  * Choice between `RepeatedStratifiedKFold` and `RobustLeavePGroupsOut` in `mriqc_clf`
  * Write cross-validation results to an `.npz` file.
* ENH: First revision of the classifier (`#553 <https://github.com/nipreps/mriqc/pull/553>`__):
  * Add the possibility of changing the scorer function.
  * Unifize labels for raters in data tables (to `rater_1`)
  * Add the possibility of setting a custom decision threshold
  * Write the probabilities in the prediction file
  * Revised `mriqc_clf` processing flow
  * Revised labels file for ds030.
  * Add IQMs for ABIDE and DS030 calculated with MRIQC 0.9.6.
* ANNOUNCEMENT: Dropped support for Python<-3.4
* WARNING (`#596 <https://github.com/nipreps/mriqc/pull/596>`__):
  We have changed the default number of threads for ANTs. Using parallelism with ANTs
  causes numerical instability on the calculated measures. The most sensitive metrics to this
  problem are the kurtosis calculations on the intensities of regions and qi_2.

0.9.6
-----

* ENH: Finished setting up `MRIQC Web API <https://mriqc.nimh.nih.gov>`_
* ENH: Better error message when --participant_label is set (`#542 <https://github.com/nipreps/mriqc/pull/542>`__)
* FIX: Allow --load-classifier option to be empty in mriqc_clf (`#544 <https://github.com/nipreps/mriqc/pull/544>`__)
* FIX: Borked bias estimation derived from Conform (`#541 <https://github.com/nipreps/mriqc/pull/541>`__)
* ENH: Test against web API 0.3.2 (`#540 <https://github.com/nipreps/mriqc/pull/540>`__)
* ENH: Change the default Web API address (`#539 <https://github.com/nipreps/mriqc/pull/539>`__)
* ENH: MRIQCWebAPI: hash fields that may have PI (`#538 <https://github.com/nipreps/mriqc/pull/538>`__)
* ENH: Added token authorization to MRIQCWebAPI client (`#535 <https://github.com/nipreps/mriqc/pull/535>`__)
* FIX: Do not mask and antsAffineInitializer twice (`#534 <https://github.com/nipreps/mriqc/pull/534>`__)
* FIX: Datasets where air (hat) mask is empty (`#533 <https://github.com/nipreps/mriqc/pull/533>`__)
* ENH: Integration testing for MRIQCWebAPI (`#520 <https://github.com/nipreps/mriqc/pull/520>`__)
* ENH: Use AFNI to calculate gcor (`#531 <https://github.com/nipreps/mriqc/pull/531>`__)
* ENH: Refactor derivatives (`#530 <https://github.com/nipreps/mriqc/pull/530>`__)
* ENH: New bold-IQM: dummy_trs (non-stady state volumes) (`#524 <https://github.com/nipreps/mriqc/pull/524>`__)
* FIX: Order of BIDS components in IQMs CSV table (`#525 <https://github.com/nipreps/mriqc/pull/525>`__)
* ENH: Improved logging of mriqc_run (`#526 <https://github.com/nipreps/mriqc/pull/526>`__)

0.9.5
-----

* ENH: Refactored structural metrics calculation (`#513 <https://github.com/nipreps/mriqc/pull/513>`__)
* ENH: Calculate rotation mask (`#515 <https://github.com/nipreps/mriqc/pull/515>`__)
* ENH: Intensity harmonization in the anatomical workflow (`#510 <https://github.com/nipreps/mriqc/pull/510>`__)
* ENH: Set N4BiasFieldCorrection number of threads (`#506 <https://github.com/nipreps/mriqc/pull/506>`__)
* ENH: Convert FWHM in pixel units (`#503 <https://github.com/nipreps/mriqc/pull/503>`__)
* ENH: Add MRIQC client for feature crowdsourcing (`#464 <https://github.com/nipreps/mriqc/pull/464>`__)
* DOC: Fix functional feature labels in documentation (docs_only) (`#507 <https://github.com/nipreps/mriqc/pull/507>`__)
* FIX: New implementation for the rPVE feature (normalization, left-tail values) (`#505 <https://github.com/nipreps/mriqc/pull/505>`__)
* ENH: Parse BIDS selectors (run, task, etc.), improve CLI (`#504 <https://github.com/nipreps/mriqc/pull/504>`__)


0.9.4
-----

* ANNOUNCEMENT: Dropped Python 2 support
* ENH: Use versioneer to handle versions (`#500 <https://github.com/nipreps/mriqc/pull/500>`__)
* ENH: Speed up spatial normalization (`#495 <https://github.com/nipreps/mriqc/pull/495>`__)
* ENH: Resampling of hat mask and TPMs with linear interp (`#498 <https://github.com/nipreps/mriqc/pull/498>`__)
* TST: Build documentation in CircleCI (`#484 <https://github.com/nipreps/mriqc/pull/484>`__)
* ENH: Use full-resolution T1w images from ABIDE (`#486 <https://github.com/nipreps/mriqc/pull/486>`__)
* TST: Parallelize tests (`#493 <https://github.com/nipreps/mriqc/pull/493>`__)
* TST: Binding /etc/localtime stopped working in docker 1.9.1 (`#492 <https://github.com/nipreps/mriqc/pull/492>`__)
* TST: Downgrade docker to 1.9.1 in circle (build_only) (`#491 <https://github.com/nipreps/mriqc/pull/491>`__)
* TST: Check for changes in intermediate nifti files (`#485 <https://github.com/nipreps/mriqc/pull/485>`__)
* FIX: Erroneous flag --n_proc in CircleCI (`#490 <https://github.com/nipreps/mriqc/pull/490>`__)
* ENH: Add build_only tag to circle builds (`#488 <https://github.com/nipreps/mriqc/pull/488>`__)
* ENH: Update Dockerfile (`#482 <https://github.com/nipreps/mriqc/pull/482>`__)
* FIX: Ignore --profile flag with Linear plugin (`#483 <https://github.com/nipreps/mriqc/pull/483>`__)
* DOC: Deep revision of the documentation (`#479 <https://github.com/nipreps/mriqc/pull/479>`__)
* ENH: Minor improvements: SpatialNormalization and segmentation (`#472 <https://github.com/nipreps/mriqc/pull/472>`__)
* ENH: Fixed typo for neurodebian install via apt-get (`#478 <https://github.com/nipreps/mriqc/pull/478>`__)
* ENH: Updating fs2gif script (`#465 <https://github.com/nipreps/mriqc/pull/465>`__)
* ENH: RF: Use niworkflows.interface.SimpleInterface (`#468 <https://github.com/nipreps/mriqc/pull/468>`__)
* ENH: Add reproducibility of metrics tracking (`#466 <https://github.com/nipreps/mriqc/pull/466>`__)

Release 0.9.3
-------------

* ENH: Reafactor of the Dockerfile to improve transparency, reduce size, and enable injecting code in Singularity (`#457 <https://github.com/nipreps/mriqc/pull/457>`__)
* ENH: Make more the memory consumption estimates of each processing step more conservative to improve robustness (`#456 <https://github.com/nipreps/mriqc/pull/456>`__)
* FIX: Minor documentation cleanups (`#461 <https://github.com/nipreps/mriqc/pull/461>`__)

Release 0.9.2
-------------

* ENH: Optional ICA reports for identifying spatiotemporal artifacts (`#412 <https://github.com/nipreps/mriqc/pull/412>`__)
* ENH: Add --profile flag (`#435 <https://github.com/nipreps/mriqc/pull/435>`__)
* ENH: Crashfiles are saved in plain text to improve portability (`#434 <https://github.com/nipreps/mriqc/pull/434>`__)
* FIX: Fixes EPI mask erosion (`#442 <https://github.com/nipreps/mriqc/pull/442>`__)
* ENH: Make FSL and AFNI motion correction more comparable by using the same scheme for defining the reference image (`#444 <https://github.com/nipreps/mriqc/pull/444>`__)
* FIX: Temporarily disabling T1w quality classifier until it can be retrained on new measures (`#447 <https://github.com/nipreps/mriqc/pull/447>`__)

Release 0.9.1
-------------

* ENH: Add mriqc version and input image hash to IQMs json file (`#432 <https://github.com/nipreps/mriqc/pull/432>`__)
* FIX: Affine and warp transforms are now applied in the correct order (`#431 <https://github.com/nipreps/mriqc/pull/431>`__)

Release 0.9.0-2
---------------

* ENH: Revise Docker paths (`#429 <https://github.com/nipreps/mriqc/pull/429>`__)
* FIX: Greedy participant selection (`#426 <https://github.com/nipreps/mriqc/pull/426>`__)
* FIX: Pin pybids to new version 0.1.0 (`#427 <https://github.com/nipreps/mriqc/pull/427>`__)
* FIX: Amends sloppy PR #425 (`#428 <https://github.com/nipreps/mriqc/pull/428>`__)

Release 0.9.0-1
---------------

* FIX: BOLD reports clipped IQMs after spikes_num (`#425 <https://github.com/nipreps/mriqc/pull/425>`__)
* FIX: Unicode error writing group reports (`#424 <https://github.com/nipreps/mriqc/pull/424>`__)
* FIX: Respect Nifi header in fMRI conform node (`#415 <https://github.com/nipreps/mriqc/pull/415>`__)
* DOC: Deep revision of documentation (#411, #416)
* ENH: Added sphinx extension to plot workflow graphs (`#411 <https://github.com/nipreps/mriqc/pull/411>`__)
* FIX: Removed repeated bias correction on anatomical workflows (`#410 <https://github.com/nipreps/mriqc/pull/410>`__)
* FIX: Race condition in bold workflow when using shared workdir (`#409 <https://github.com/nipreps/mriqc/pull/409>`__)
* FIX: Tests (#408, #407, #405)
* FIX: Remove CDN for group level reports (`#406 <https://github.com/nipreps/mriqc/pull/406>`__)
* FIX: Unused connection, matplotlib segfault (#403, #402)
* ENH: Skip SpikeFFT detector by default (`#400 <https://github.com/nipreps/mriqc/pull/400>`__)
* ENH: Use float32 (`#399 <https://github.com/nipreps/mriqc/pull/399>`__)
* ENH: Spike finder performance improvoments (`#398 <https://github.com/nipreps/mriqc/pull/398>`__)
* ENH: Basic T2w workflow (`#394 <https://github.com/nipreps/mriqc/pull/394>`__)
* ENH: Re-enable 3dvolreg (`#390 <https://github.com/nipreps/mriqc/pull/390>`__)
* ENH: Add T1w classifier (`#389 <https://github.com/nipreps/mriqc/pull/389>`__)

Release 0.9.0-0
---------------

* FIX: Remove non-repeatable step from pipeline (`#369 <https://github.com/nipreps/mriqc/pull/369>`__)
* ENH: Improve group level command line, with more informative output when no IQMs are found for a modality (`#372 <https://github.com/nipreps/mriqc/pull/372>`__)
* ENH: Make group reports self-contained (`#333 <https://github.com/nipreps/mriqc/pull/333>`__)
* FIX: New mosaics, based on old ones (#361, #360, #334)
* FIX: Require numpy>=1.12 to avoid casting problems (`#356 <https://github.com/nipreps/mriqc/pull/356>`__)
* FIX: Add support for acq and rec tags of BIDS (`#346 <https://github.com/nipreps/mriqc/pull/346>`__)
* DOC: Documentation updates (`#350 <https://github.com/nipreps/mriqc/pull/350>`__)
* FIX: pybids compatibility "No scans were found" (#340, #347, #342)
* ENH: Rewrite PYTHONPATH in docker/singularity images (`#345 <https://github.com/nipreps/mriqc/pull/345>`__)
* ENH: Move metadata onto the bottom of the individual reports (`#332 <https://github.com/nipreps/mriqc/pull/332>`__)
* ENH: Don't include MNI registration report unlesS --verbose-reports is used (`#362 <https://github.com/nipreps/mriqc/pull/362>`__)


Release 0.8.9
-------------

* ENH: Added registration svg panel to reports (`#297 <https://github.com/nipreps/mriqc/pull/297>`__)


Release 0.8.8
-------------

* FIX: Bug translating int16 to uint8 in conform image.
* FIX: Error in ConformImage interface (`#297 <https://github.com/nipreps/mriqc/pull/297>`__)
* ENH: Replace BBR by ANTs (#295, #296)
* FIX: Singularity: user-environment leaking into container (`#293 <https://github.com/nipreps/mriqc/pull/293>`__)
* ENH: Report failed cases in group report (`#291 <https://github.com/nipreps/mriqc/pull/291>`__)
* FIX: Brighter anatomical --verbose-reports (`#290 <https://github.com/nipreps/mriqc/pull/290>`__)
* FIX: X-flip in the mosaics (`#289 <https://github.com/nipreps/mriqc/pull/289>`__)
* ENH: Show metadata in the individual report (`#288 <https://github.com/nipreps/mriqc/pull/288>`__)
* ENH: Label in the cutoff threshold - fmriplot (`#287 <https://github.com/nipreps/mriqc/pull/287>`__)
* ENH: PyBIDS (`#286 <https://github.com/nipreps/mriqc/pull/286>`__)
* ENH: Simplify tests (`#284 <https://github.com/nipreps/mriqc/pull/284>`__)
* FIX: MRIQC crashed generating csv files (`#283 <https://github.com/nipreps/mriqc/pull/283>`__)
* FIX: Bug in setup.py (`#281 <https://github.com/nipreps/mriqc/pull/281>`__)
* ENH: Makefile (`#280 <https://github.com/nipreps/mriqc/pull/280>`__)
* FIX: Revision of IQMs (#266, #272, #279)
* ENH: Deprecation of --nthreads, new flags (`#260 <https://github.com/nipreps/mriqc/pull/260>`__)
* ENH: Improvements on plots rendering (#254, #257, #258, #267, #268, #269, #270)
* ENH: FFT detection of spikes (#253, #272)
* FIX: Labels and links of samples in group plots (`#249 <https://github.com/nipreps/mriqc/pull/249>`__)
* ENH: Units in group plots (`#242 <https://github.com/nipreps/mriqc/pull/242>`__)
* FIX: More reliable group level (`#238 <https://github.com/nipreps/mriqc/pull/238>`__)
* ENH: Add --verbose-reports for fMRI (`#236 <https://github.com/nipreps/mriqc/pull/236>`__)
* ENH: Migrate functional reports to html (`#232 <https://github.com/nipreps/mriqc/pull/232>`__)
* ENH: Add 0.2 FD cutoff line (`#231 <https://github.com/nipreps/mriqc/pull/231>`__)
* ENH: Add AFNI's outlier count to carpet plot confound charts (`#230 <https://github.com/nipreps/mriqc/pull/230>`__)

Release 0.8.7
-------------

* ENH: Anatomical Group reports in html (`#227 <https://github.com/nipreps/mriqc/pull/227>`__)
* ENH: Add kurtosis to summary statistics (`#224 <https://github.com/nipreps/mriqc/pull/224>`__)
* ENH: New report layout for fMRI, added carpetplot (`#198 <https://github.com/nipreps/mriqc/pull/198>`__)
* ENH: Anatomical workflow refactor (`#219 <https://github.com/nipreps/mriqc/pull/219>`__).

Release 0.8.6
-------------

* [FIX, CRITICAL] Do not chmod in Docker internal scripts
* FIX: Error creating derivatives folder
* ENH: Moved MNI spatial normalization to NIworkflows, and made robust.
* ENH: De-coupled participant and group (reports) levels
* ENH: Use new FD and DVARs calculations from nipype (`#172 <https://github.com/nipreps/mriqc/pull/172>`__)
* ENH: Started with python3 compatibility
* ENH: Added new M2WM measure #158
* FIX: QI2 is skipped if background intensity is not appropriate (`#147 <https://github.com/nipreps/mriqc/pull/147>`__)

Release 0.8.5
-------------

* FIX: Error inverting the T1w-to-MNI warping (`#146 <https://github.com/nipreps/mriqc/pull/146>`__)
* FIX: TypeError computing DVARS (`#145 <https://github.com/nipreps/mriqc/pull/145>`__)
* ENH: Plot figure of fitted background chi for QI2 (`#143 <https://github.com/nipreps/mriqc/pull/143>`__)
* ENH: Move skull-stripping and reorient to NIworkflows (`#142 <https://github.com/nipreps/mriqc/pull/142>`__)
* FIX: mriqc crashes if no anatomical scans are found (`#141 <https://github.com/nipreps/mriqc/pull/141>`__)
* DOC: Added acknowledgments to CPAC team members (`#134 <https://github.com/nipreps/mriqc/pull/134>`__)
* ENH: Use absolute imports (`#133 <https://github.com/nipreps/mriqc/pull/133>`__)
* FIX: VisibleDeprecationWarning (`#132 <https://github.com/nipreps/mriqc/pull/132>`__)
* ENH: Provide full FD/DVARS files (`#128 <https://github.com/nipreps/mriqc/pull/128>`__)
* ENH: Use MCFLIRT to compute motion parameters. AFNI's 3dvolreg now is optional (`#121 <https://github.com/nipreps/mriqc/pull/121>`__)
* FIX: BIDS trees with anatomical images with different acquisition tokens (`#116 <https://github.com/nipreps/mriqc/pull/116>`__)
* FIX: BIDS trees with anatomical images with several runs (`#112 <https://github.com/nipreps/mriqc/pull/112>`__)
* ENH: Options for ANTs normalization: reduced test times (`#124 <https://github.com/nipreps/mriqc/pull/124>`__),
  and updated options (`#115 <https://github.com/nipreps/mriqc/pull/115>`__)

Release 0.8.4
-------------

* ENH: PDF reports now use RST templates and jinja2 (`#109 <https://github.com/nipreps/mriqc/pull/109>`__)
* FIX: Single-session-multiple-run anatomical files were not correctly located (`#112 <https://github.com/nipreps/mriqc/pull/112>`__)

Release 0.8.3
-------------

* DOC: Added examples of the PDF reports (`#107 <https://github.com/nipreps/mriqc/pull/107>`__)
* FIX: Fixed problems with Python 3 when generating reports.

Release 0.8.2
-------------

* ENH: Python 3 compatibility (`#99 <https://github.com/nipreps/mriqc/pull/99>`__)
* ENH: Add JSON settings file for ANTS (`#95 <https://github.com/nipreps/mriqc/pull/95>`__)
* ENH: Generate reports automatically if mriqc is run without the -S flag (`#93 <https://github.com/nipreps/mriqc/pull/93>`__)
* FIX: Revised implementation of QI2 measure (`#90 <https://github.com/nipreps/mriqc/pull/90>`__)
* AGAVE: Fixed docker image for agave (`#89 <https://github.com/nipreps/mriqc/pull/89>`__)
* FIX: Problem when generating the air mask with dipy installed (`#88 <https://github.com/nipreps/mriqc/pull/88>`__)
* ENH: One-session-one-run execution mode (`#85 <https://github.com/nipreps/mriqc/pull/85>`__)
* AGAVE: Added an agave app description generator (`#84 <https://github.com/nipreps/mriqc/pull/84>`__)

Release 0.3.0
-------------

* ENH: Updated CircleCI and Docker to use the version 2.1.0 of ANTs
  compiled by their developers.
* ENH: New anatomical workflows to compute the air mask (`#56 <https://github.com/nipreps/mriqc/pull/56>`__)

Release 0.1.0
-------------

* FIX: #55
* ENH: Added rotation of output csv files if they exist

Release 0.0.2
-------------

* ENH: Completed migration from QAP
* ENH: Integration with ReadTheDocs
* ENH: Submission to PyPi

Release 0.0.1
-------------

* Basic mriqc functionality
