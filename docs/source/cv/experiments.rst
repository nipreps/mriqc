
.. _experiment0:

Experiment 0: What makes a good cross-validation split in this application?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We first wanted to understand what kind of data split we should use to benchmark
the results we get using the classifier.

In a very simplistic approach, we use ``mriqc_clf`` to check if there are differences
between our :abbr:`LoSo (leave-one-site-out)` splits or a sandard 10-fold.
For that, we use the arguments ``--nested_cv`` and ``--nested_cv_kfold`` respectively.
Please note we are using the ``--debug`` flag to reduce the number of hyperparameters
tested in the inner cross-validation loop. Here we are cross-validating the performance
of the model selection technique: whether we will use :abbr:`LoSo (leave-one-site-out)`
or 10-fold.

Running:

  ::

      docker run -ti -v $PWD:/scratch -w /scratch \
                 --entrypoint=/usr/local/miniconda/bin/mriqc_clf \
                 poldracklab/mriqc:0.9.7 \
                 --train --test --log-file --nested_cv_kfold --cv kfold -v --debug

We get the following output (filtered):

  ::

      Nested CV [avg] roc_auc=0.869013861525, accuracy=0.82746147315
      Nested CV roc_auc=0.925, 0.881, 0.764, 0.904, 0.840, 0.864, 0.883, 0.857, 0.865, 0.909.
      Nested CV accuracy=0.847, 0.874, 0.757, 0.838, 0.773, 0.855, 0.809, 0.835, 0.826, 0.862.
      ...
      CV [Best model] roc_auc=0.855578059459, mean=0.856, std=0.002.
      ...
      Ratings distribution: 190/75 (71.70%/28.30%, accept/exclude)
      Testing on evaluation (left-out) dataset ...
      Performance on evaluation set (roc_auc=0.677, accuracy=0.747)
      Predictions: 253 (accept) / 12 (exclude)
      Classification report:
                   precision    recall  f1-score   support
           accept       0.74      0.99      0.85       190
          exclude       0.83      0.13      0.23        75
      avg / total       0.77      0.75      0.67       265


Please note that the outer loop evaluated an average AUC of 0.87, an average accuracy of 83%.
Then, fitting the model (using only the inner cross-validation loop on the whole dataset)
yielded an AUC=0.85 with very small variability. However, when evaluated on the held-out
dataset, the AUC dropped to 0.68 and the accuracy to 75%.

Let's repeat the experiment, but using :abbr:`LoSo (leave-one-site-out)` in the inner loop:

  ::

      docker run -ti -v $PWD:/scratch -w /scratch \
                 --entrypoint=/usr/local/miniconda/bin/mriqc_clf \
                 poldracklab/mriqc:0.9.7 \
                 --train --test --log-file --nested_cv_kfold --cv loso -v --debug


We get the following output (filtered):

  ::

      Nested CV [avg] roc_auc=0.858722005549, accuracy=0.819287243874
      Nested CV roc_auc=0.908, 0.874, 0.761, 0.914, 0.826, 0.842, 0.871, 0.850, 0.835, 0.906.
      Nested CV accuracy=0.838, 0.865, 0.739, 0.829, 0.782, 0.827, 0.827, 0.807, 0.817, 0.862.
      ...
      CV [Best model] roc_auc=0.744096956862, mean=0.744, std=0.112.
      ...
      Ratings distribution: 190/75 (71.70%/28.30%, accept/exclude)
      Testing on evaluation (left-out) dataset ...
      Performance on evaluation set (roc_auc=0.706, accuracy=0.770)
      Predictions: 247 (accept) / 18 (exclude)
      Classification report:
                   precision    recall  f1-score   support

           accept       0.76      0.99      0.86       190
          exclude       0.89      0.21      0.34        75

      avg / total       0.80      0.77      0.71       265


Therefore, we see that using 10-fold for the split of the outer cross-validation loop, gives us
an average AUC of 0.86 and an accuracy of 82%. Below, we see the results of fitting that model.
In a cross-validation using :abbr:`LoSo (leave-one-site-out)`, the AUC drops to 0.744. Finally
if we test the model on our left-out dataset, the final AUC is 0.71 and the accuracy 77%.

Two more evaluations, now using :abbr:`LoSo (leave-one-site-out)` in the outer loop:

  ::

      docker run -ti -v $PWD:/scratch -w /scratch \
                 --entrypoint=/usr/local/miniconda/bin/mriqc_clf \
                 poldracklab/mriqc:0.9.7 \
                 --train --test --log-file --nested_cv --cv kfold -v --debug


  ::

      Nested CV [avg] roc_auc=0.710537391399, accuracy=0.759618741224
      Nested CV roc_auc=0.780, 0.716, 0.829, 0.877, 0.391, 0.632, 0.679, 0.634, 0.665, 0.472, 0.690, 0.963, 0.917, 0.528, 0.813, 0.743, 0.751.
      Nested CV accuracy=0.796, 0.421, 0.869, 0.583, 0.852, 0.625, 0.807, 0.767, 0.703, 0.357, 0.832, 0.964, 0.911, 0.947, 0.750, 0.870, 0.860.
      ...
      CV [Best model] roc_auc=0.872377212756, mean=0.872, std=0.019.
      ...
      Ratings distribution: 190/75 (71.70%/28.30%, accept/exclude)
      Testing on evaluation (left-out) dataset ...
      Performance on evaluation set (roc_auc=0.685, accuracy=0.762)
      Predictions: 249 (accept) / 16 (exclude)
      Classification report:
                   precision    recall  f1-score   support

           accept       0.76      0.99      0.86       190
          exclude       0.88      0.19      0.31        75

      avg / total       0.79      0.76      0.70       265


And finally :abbr:`LoSo (leave-one-site-out)` in both outer and inner loops:

  ::

      docker run -ti -v $PWD:/scratch -w /scratch \
                 --entrypoint=/usr/local/miniconda/bin/mriqc_clf \
                 poldracklab/mriqc:0.9.7 \
                 --train --test --log-file --nested_cv --cv loso -v --debug

  ::

      Nested CV [avg] roc_auc=0.715716013846, accuracy=0.752136647911
      Nested CV roc_auc=0.963, 0.756, 0.554, 0.685, 0.673, 0.659, 0.584, 0.764, 0.787, 0.764, 0.883, 0.843, 0.846, 0.431, 0.599, 0.910, 0.465.
      Nested CV accuracy=0.964, 0.898, 0.852, 0.789, 0.531, 0.821, 0.767, 0.947, 0.722, 0.842, 0.528, 0.778, 0.869, 0.357, 0.766, 0.931, 0.425.
      ...
      CV [Best model] roc_auc=0.712039797411, mean=0.712, std=0.124.
      ...
      Ratings distribution: 190/75 (71.70%%/28.30%%, accept/exclude)
      Testing on evaluation (left-out) dataset ...
      Performance on evaluation set (roc_auc=0.685, accuracy=0.766)
      Predictions: 244 (accept) / 21 (exclude)
      Classification report:
                   precision    recall  f1-score   support

           accept       0.76      0.98      0.86       190
          exclude       0.81      0.23      0.35        75

      avg / total       0.78      0.77      0.71       265


Using :abbr:`LoSo (leave-one-site-out)` in the outer loop the average AUC is not that optimistic
(0.78 using K-Fold in the inner loop and 0.71 using :abbr:`LoSo (leave-one-site-out)`). Same
stands for average accuracy (76%/75% K-Fold/:abbr:`LoSo (leave-one-site-out)` in the inner loop).

When checking these results with respect to the performance on the held out dataset, the main
interpretation that arises is that the 10-Fold cross-validation is overestimating the performance.
The features have an structure correlated with the site of origin, and the 10-Fold splits do not
represent that structure well. All the folds learn something about all sites, and thus, this
cross-validated result cannot be considered a good estimation of performance on data from unseen
sites.
