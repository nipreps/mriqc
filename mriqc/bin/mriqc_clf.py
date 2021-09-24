# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# Copyright 2021 The NiPreps Developers <nipreps@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# We support and encourage derived works from this project, please read
# about our expectations at
#
#     https://www.nipreps.org/community/licensing/
#
"""
``mriqc_fit`` command line interface definition.
"""
import logging
import sys
import warnings
from argparse import ArgumentParser, Namespace, RawTextHelpFormatter
from os.path import abspath, isfile

import matplotlib
from mriqc.bin import messages
from pkg_resources import resource_filename as pkgrf

matplotlib.use("Agg")

try:
    from sklearn.metrics.base import UndefinedMetricWarning
except ImportError:
    from sklearn.exceptions import UndefinedMetricWarning

BASE_NAME = "mclf_run-{time}_mod-{model}_ver-{version}_class-{multiclass}_cv-{cv}"
CLASSIFIER_PICKLE = (
    "mriqc",
    "data/mclf_run-20170724-191452_mod-rfc_ver-0.9.7-rc8_class-2_cv-loso_data-all_estimator.pklz",
)
TIME_FORMAT = "%Y%m%d-%H%M%S"

# Logger setup
LOG_FORMAT = "%(asctime)s %(name)s:%(levelname)s %(message)s"
warnings.simplefilter("once", UndefinedMetricWarning)
LOGGER = logging.getLogger("mriqc.classifier")
_handler = logging.StreamHandler(stream=sys.stdout)
_handler.setFormatter(logging.Formatter(fmt=LOG_FORMAT, datefmt="%y%m%d-%H:%M:%S"))
LOGGER.addHandler(_handler)

cached_warnings = []


def warn_redirect(
    message: str,
    category: str,
    filename: str,
    lineno: int,
    file: str = None,
    line: str = None,
) -> None:
    """
    Caches a list of raised warning categories.

    Parameters
    ----------
    message : str
        Raised warning message
    category : str
        Raised warning message category
    filename : str
        Name of the file for which the warning was raised
    lineno : int
        Line number for which the warning was raised
    file : str, optional
        Path of the file for which the warning was raised, by default None
    line : int, optional
        Line content for which the warning was raised, by default None
    """
    if category not in cached_warnings:
        # Log captured warning
        debug_message = messages.CLF_CAPTURED_WARNING.format(
            category=category, message=message
        )
        LOGGER.debug(debug_message)
        # Add category to cache
        cached_warnings.append(category)


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(
        description="MRIQC model selection and held-out evaluation.",
        formatter_class=RawTextHelpFormatter,
    )

    g_clf = parser.add_mutually_exclusive_group()
    g_clf.add_argument(
        "--train",
        nargs="*",
        help="training data tables, X and Y, leave empty for ABIDE",
    )
    g_clf.add_argument(
        "--load-classifier",
        nargs="?",
        type=str,
        default="",
        help="load a previously saved classifier",
    )

    parser.add_argument(
        "--test", nargs="*", help="test data tables, X and Y, leave empty for DS030"
    )
    parser.add_argument(
        "-X", "--evaluation-data", help="classify this CSV table of IQMs"
    )

    parser.add_argument(
        "--train-balanced-leaveout",
        action="store_true",
        default=False,
        help="leave out a balanced, random, sample of training examples",
    )
    parser.add_argument(
        "--multiclass",
        "--ms",
        action="store_true",
        default=False,
        help="do not binarize labels",
    )

    g_input = parser.add_argument_group("Options")
    g_input.add_argument("-P", "--parameters", action="store")
    g_input.add_argument(
        "-M",
        "--model",
        action="store",
        default="rfc",
        choices=["rfc", "xgb", "svc_lin", "svc_rbf"],
        help="model under test",
    )
    g_input.add_argument(
        "--nested_cv",
        action="store_true",
        default=False,
        help="run nested cross-validation before held-out",
    )
    g_input.add_argument(
        "--nested_cv_kfold",
        action="store_true",
        default=False,
        help="run nested cross-validation before held-out, "
        "using 10-fold split in the outer loop",
    )
    g_input.add_argument(
        "--perm",
        action="store",
        default=0,
        type=int,
        help="permutation test: number of permutations",
    )

    g_input.add_argument("-S", "--scorer", action="store", default="roc_auc")
    g_input.add_argument(
        "--cv",
        action="store",
        default="loso",
        choices=["kfold", "loso", "balanced-kfold", "batch"],
    )
    g_input.add_argument("--debug", action="store_true", default=False)

    g_input.add_argument(
        "--log-file",
        nargs="?",
        action="store",
        default="",
        help="write log to this file, leave empty for a default log name",
    )

    g_input.add_argument(
        "-v",
        "--verbose",
        dest="verbose_count",
        action="count",
        default=0,
        help="increases log verbosity for each occurence.",
    )
    g_input.add_argument(
        "--njobs", action="store", default=-1, type=int, help="number of jobs"
    )

    g_input.add_argument(
        "-t",
        "--threshold",
        action="store",
        default=0.5,
        type=float,
        help="decision threshold of the classifier",
    )

    return parser


def get_log_level(opts: Namespace) -> int:
    if opts.verbose_count > 1:
        return int(max(25 - 5 * opts.verbose_count, 1))
    else:
        return int(max(3 - opts.verbose_count, 0) * 10)


def main():
    """Entry point."""
    import re
    from datetime import datetime

    from .. import __version__
    from ..classifier.helper import CVHelper

    warnings.showwarning = warn_redirect

    opts = get_parser().parse_args()

    log_level = get_log_level(opts)
    LOGGER.setLevel(log_level)

    time = datetime.now().strftime(TIME_FORMAT)
    version = re.sub(r"[\+_@]", ".", __version__)
    multiclass = 3 if opts.multiclass else 2
    base_name = BASE_NAME.format(
        time=time,
        model=opts.model,
        version=version,
        multiclass=multiclass,
        cv=opts.cv,
    )

    if opts.nested_cv_kfold:
        base_name += "_ncv-kfold"
    elif opts.nested_cv:
        base_name += "_ncv-loso"

    if opts.log_file is None or len(opts.log_file) > 0:
        log_file = opts.log_file if opts.log_file else base_name + ".log"
        fhl = logging.FileHandler(log_file)
        fhl.setFormatter(fmt=logging.Formatter(LOG_FORMAT))
        fhl.setLevel(log_level)
        LOGGER.addHandler(fhl)

    clf_loaded = False

    if opts.train is not None:
        # Initialize model selection helper
        train_path = _parse_set(opts.train, default="abide")
        cv_helper = CVHelper(
            X=train_path[0],
            Y=train_path[1],
            n_jobs=opts.njobs,
            scorer=opts.scorer,
            b_leaveout=opts.train_balanced_leaveout,
            multiclass=opts.multiclass,
            verbosity=opts.verbose_count,
            split=opts.cv,
            model=opts.model,
            debug=opts.debug,
            basename=base_name,
            nested_cv=opts.nested_cv,
            nested_cv_kfold=opts.nested_cv_kfold,
            param_file=opts.parameters,
            permutation_test=opts.perm,
        )

        if opts.cv == "batch" or opts.perm:
            test_path = _parse_set(opts.test, default="ds030")
            # Do not set x_test unless we are going to run batch exp.
            cv_helper.setXtest(test_path[0], test_path[1])

        # Perform model selection before setting held-out data, for hygene
        cv_helper.fit()

        # Pickle if required
        cv_helper.save(suffix="data-train_estimator")

    # If no training set is given, need a classifier
    else:
        load_classifier = opts.load_classifier
        if load_classifier is None:
            load_classifier = pkgrf(*CLASSIFIER_PICKLE)
        if not isfile(load_classifier):
            info = (
                "was not provided"
                if load_classifier == ""
                else f"('{load_classifier}') was not found"
            )
            runtime_error = messages.CLF_CLASSIFIER_MISSING.format(info=info)
            raise RuntimeError(runtime_error)

        cv_helper = CVHelper(
            load_clf=load_classifier,
            n_jobs=opts.njobs,
            rate_label=["rater_1"],
            basename=base_name,
        )
        clf_loaded = True

    test_path = _parse_set(opts.test, default="ds030")
    if test_path and opts.cv != "batch":
        # Set held-out data
        cv_helper.setXtest(test_path[0], test_path[1])
        # Evaluate
        cv_helper.evaluate(
            matrix=True, scoring=[opts.scorer, "accuracy"], save_pred=True
        )

        # Pickle if required
        if not clf_loaded:
            cv_helper.fit_full()
            cv_helper.save(suffix="data-all_estimator")

    if opts.evaluation_data:
        cv_helper.predict_dataset(
            opts.evaluation_data, save_pred=True, thres=opts.threshold
        )

    results_path = abspath(cv_helper._base_name + "*")
    saved_results_message = messages.CLF_SAVED_RESULTS.format(path=results_path)
    LOGGER.info(saved_results_message)


def _parse_set(arg, default):
    if arg is not None and len(arg) == 0:
        names = f"x_{default}.csv", f"y_{default}.csv"
        return [pkgrf("mriqc", f"data/csv/{name}") for name in names]

    if arg is not None and len(arg) not in (0, 2):
        raise RuntimeError(messages.CLF_WRONG_PARAMETER_COUNT)

    if arg is None:
        return None

    if len(arg) == 2:
        train_exists = [isfile(fname) for fname in arg]
        if len(train_exists) > 0 and not all(train_exists):
            errors = [
                f"file '{fname}' not found"
                for fexists, fname in zip(train_exists, arg)
                if not fexists
            ]
            runtime_error = messages.CLF_TRAIN_LOAD_ERROR.format(
                n_errors=len(errors), errors=", ".join(errors)
            )
            raise RuntimeError(runtime_error)
    return arg


if __name__ == "__main__":
    main()
