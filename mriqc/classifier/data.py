# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
===================
Data handler module
===================

Reads in and writes CSV files with the IQMs


"""

from pathlib import Path
import numpy as np
import pandas as pd
from builtins import str

from .. import config
from ..utils.misc import BIDS_COMP


def get_groups(X, label="site"):
    """Generate the index of sites"""
    groups = X[label].values.ravel().tolist()
    gnames = sorted(list(set(groups)))
    return [gnames.index(g) for g in groups], gnames


def combine_datasets(inputs, rating_label="rater_1"):
    mdata = []
    for dataset_x, dataset_y, sitename in inputs:
        sitedata, _ = read_dataset(
            dataset_x,
            dataset_y,
            rate_label=rating_label,
            binarize=True,
            site_name=sitename,
        )
        sitedata["database"] = [sitename] * len(sitedata)

        if "site" not in sitedata.columns.ravel().tolist():
            sitedata["site"] = [sitename] * len(sitedata)

        mdata.append(sitedata)

    mdata = pd.concat(mdata)

    all_cols = mdata.columns.ravel().tolist()

    bids_comps = list(BIDS_COMP.keys())
    bids_comps_present = list(set(mdata.columns.ravel().tolist()) & set(bids_comps))
    bids_comps_present = [bit for bit in bids_comps if bit in bids_comps_present]

    ordered_cols = bids_comps_present + ["database", "site", "rater_1"]
    ordered_cols += sorted(list(set(all_cols) - set(ordered_cols)))
    return mdata[ordered_cols]


def get_bids_cols(dataframe):
    """ Returns columns corresponding to BIDS bits """
    bids_comps = list(BIDS_COMP.keys())
    bids_comps_present = list(set(dataframe.columns.ravel().tolist()) & set(bids_comps))
    return [bit for bit in bids_comps if bit in bids_comps_present]


def read_iqms(feat_file):
    """ Reads in the features """
    feat_file = Path(feat_file)

    if feat_file.suffix == ".csv":
        bids_comps = list(BIDS_COMP.keys())
        x_df = pd.read_csv(
            feat_file, index_col=False, dtype={col: str for col in bids_comps}
        )
        # Find present bids bits and sort by them
        bids_comps_present = list(set(x_df.columns.ravel().tolist()) & set(bids_comps))
        bids_comps_present = [bit for bit in bids_comps if bit in bids_comps_present]
        x_df = x_df.sort_values(by=bids_comps_present)
        # Remove sub- prefix in subject_id
        x_df.subject_id = x_df.subject_id.str.lstrip("sub-")

        # Remove columns that are not IQMs
        feat_names = list(x_df._get_numeric_data().columns.ravel())
        for col in bids_comps:
            try:
                feat_names.remove(col)
            except ValueError:
                pass
    else:
        bids_comps_present = ["subject_id"]
        x_df = pd.read_csv(
            feat_file, index_col=False, sep="\t", dtype={"bids_name": str}
        )
        x_df = x_df.sort_values(by=["bids_name"])
        x_df["subject_id"] = x_df.bids_name.str.lstrip("sub-")
        x_df = x_df.drop(columns=["bids_name"])
        x_df.subject_id = ["_".join(v.split("_")[:-1]) for v in x_df.subject_id.ravel()]
        feat_names = list(x_df._get_numeric_data().columns.ravel())

    for col in feat_names:
        if col.startswith(("size_", "spacing_", "Unnamed")):
            feat_names.remove(col)

    return x_df, feat_names, bids_comps_present


def read_labels(
    label_file,
    rate_label="rater_1",
    binarize=True,
    site_name=None,
    rate_selection="random",
    collapse=True,
):
    """
    Reads in the labels. Massage labels table to have the
    appropriate format
    """

    if isinstance(rate_label, str):
        rate_label = [rate_label]
    output_labels = rate_label

    bids_comps = list(BIDS_COMP.keys())
    y_df = pd.read_csv(
        label_file, index_col=False, dtype={col: str for col in bids_comps}
    )

    # Find present bids bits and sort by them
    bids_comps_present = get_bids_cols(y_df)
    y_df = y_df.sort_values(by=bids_comps_present)
    y_df.subject_id = y_df.subject_id.str.lstrip("sub-")
    y_df[rate_label] = y_df[rate_label].apply(pd.to_numeric, errors="raise")

    if len(rate_label) == 2:
        np.random.seed(42)
        ratermask_1 = ~np.isnan(y_df[[rate_label[0]]].values.ravel())
        ratermask_2 = ~np.isnan(y_df[[rate_label[1]]].values.ravel())

        all_rated = ratermask_1 & ratermask_2
        mergey = np.array(y_df[[rate_label[0]]].values.ravel().tolist())
        mergey[ratermask_2] = y_df[[rate_label[1]]].values.ravel()[ratermask_2]

        subsmpl = np.random.choice(
            np.where(all_rated)[0], int(0.5 * np.sum(all_rated)), replace=False
        )
        all_rated[subsmpl] = False
        mergey[all_rated] = y_df[[rate_label[0]]].values.ravel()[all_rated]
        y_df["merged_ratings"] = mergey.astype(int)

        # Set default name
        if collapse:
            cols = [
                ("indv_%s" % c) if c.startswith("rater") else c
                for c in y_df.columns.ravel().tolist()
            ]
            cols[y_df.columns.get_loc("merged_ratings")] = rate_label[0]
            y_df.columns = cols
            output_labels = [rate_label[0]]
        else:
            output_labels = rate_label
            output_labels.insert(0, "merged_ratings")

    if binarize:
        mask = y_df[output_labels[0]] >= 0
        y_df.loc[mask, output_labels[0]] = 0
        y_df.loc[~mask, output_labels[0]] = 1

    if "site" in y_df.columns.ravel().tolist():
        output_labels.insert(0, "site")
    elif site_name is not None:
        y_df["site"] = [site_name] * len(y_df)
        output_labels.insert(0, "site")

    return y_df[bids_comps_present + output_labels]


def read_dataset(
    feat_file,
    label_file,
    merged_name=None,
    binarize=True,
    site_name=None,
    rate_label="rater_1",
    rate_selection="random",
):
    """ Reads in the features and labels """

    x_df, feat_names, _ = read_iqms(feat_file)
    y_df = read_labels(
        label_file,
        rate_label,
        binarize,
        collapse=True,
        site_name=site_name,
        rate_selection=rate_selection,
    )
    if isinstance(rate_label, (list, tuple)):
        rate_label = rate_label[0]

    # Find present bids bits and sort by them
    bids_comps = list(BIDS_COMP.keys())
    bids_comps_x = list(set(x_df.columns.ravel().tolist()) & set(bids_comps))
    bids_comps_x = [bit for bit in bids_comps if bit in bids_comps_x]
    bids_comps_y = list(set(x_df.columns.ravel().tolist()) & set(bids_comps))
    bids_comps_y = [bit for bit in bids_comps if bit in bids_comps_y]

    if bids_comps_x != bids_comps_y:
        raise RuntimeError("Labels and features cannot be merged")

    x_df["bids_ids"] = x_df.subject_id.values.copy()
    y_df["bids_ids"] = y_df.subject_id.values.copy()

    for comp in bids_comps_x[1:]:
        x_df["bids_ids"] = x_df.bids_ids.str.cat(x_df.loc[:, comp].astype(str), sep="_")
        y_df["bids_ids"] = y_df.bids_ids.str.cat(y_df.loc[:, comp].astype(str), sep="_")

    # Remove failed cases from Y, append new columns to X
    y_df = y_df[y_df["bids_ids"].isin(list(x_df.bids_ids.values.ravel()))]

    # Drop indexing column
    del x_df["bids_ids"]
    del y_df["bids_ids"]

    # Merge Y dataframe into X
    x_df = pd.merge(x_df, y_df, on=bids_comps_x, how="left")

    if merged_name is not None:
        x_df.to_csv(merged_name, index=False)

    # Drop samples with invalid rating
    nan_labels = x_df[x_df[rate_label].isnull()].index.ravel().tolist()
    if nan_labels:
        config.loggers.interface.info(
            f"Dropping {len(nan_labels)} samples for having non-numerical labels,"
        )
        x_df = x_df.drop(nan_labels)

    # Print out some info
    nsamples = len(x_df)
    config.loggers.interface.info(
        f'Created dataset X="{feat_file}", Y="{label_file}" (N={nsamples} valid samples)'
    )

    # Inform about ratings distribution
    labels = sorted(list(set(x_df[rate_label].values.ravel().tolist())))
    ldist = []
    for l in labels:
        ldist.append(int(np.sum(x_df[rate_label] == l)))

    config.loggers.interface.info(
        "Ratings distribution: %s (%s, %s)",
        "/".join(["%d" % x for x in ldist]),
        "/".join(["%.2f%%" % (100 * x / nsamples) for x in ldist]),
        "accept/exclude" if len(ldist) == 2 else "exclude/doubtful/accept",
    )

    return x_df, feat_names


def balanced_leaveout(dataframe, site_column="site", rate_label="rater_1"):
    sites = list(set(dataframe[[site_column]].values.ravel()))
    pos_draw = []
    neg_draw = []

    for site in sites:
        site_x = dataframe.loc[dataframe[site_column].str.contains(site)]
        site_x_pos = site_x[site_x[rate_label] == 1]

        if len(site_x_pos) > 4:
            pos_draw.append(np.random.choice(site_x_pos.index.tolist()))

            site_x_neg = site_x[site_x[rate_label] == 0]
            neg_draw.append(np.random.choice(site_x_neg.index.tolist()))

    left_out = dataframe.iloc[pos_draw + neg_draw].copy()
    dataframe = dataframe.drop(dataframe.index[pos_draw + neg_draw])
    return dataframe, left_out


def zscore_dataset(dataframe, excl_columns=None, by="site", njobs=-1):
    """ Returns a dataset zscored by the column given as argument """
    from multiprocessing import Pool, cpu_count

    config.loggers.interface.info("z-scoring dataset ...")

    if njobs <= 0:
        njobs = cpu_count()

    sites = list(set(dataframe[[by]].values.ravel().tolist()))
    columns = list(dataframe.select_dtypes([np.number]).columns.ravel())

    if excl_columns is None:
        excl_columns = []

    for col in columns:
        if not np.isfinite(np.sum(dataframe[[col]].values.ravel())):
            excl_columns.append(col)

    if excl_columns:
        for col in excl_columns:
            try:
                columns.remove(col)
            except ValueError:
                pass

    zs_df = dataframe.copy()

    pool = Pool(njobs)
    args = [(zs_df, columns, s) for s in sites]
    results = pool.map(zscore_site, args)
    for site, res in zip(sites, results):
        zs_df.loc[zs_df.site == site, columns] = res

    zs_df.replace([np.inf, -np.inf], np.nan)
    nan_columns = zs_df.columns[zs_df.isnull().any()].tolist()

    if nan_columns:
        config.loggers.interface.warning(
            f'Columns {", ".join(nan_columns)} contain NaNs after z-scoring.'
        )
        zs_df[nan_columns] = dataframe[nan_columns].values

    return zs_df


def zscore_site(args):
    """ z-scores only one site """
    from scipy.stats import zscore

    dataframe, columns, site = args
    return zscore(dataframe.loc[dataframe.site == site, columns].values, ddof=1, axis=0)
