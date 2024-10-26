import boto3
import logging
import io
import numpy as np
import os
import pandas as pd
import pickle
import sklearn.metrics as metrics
from argparse import ArgumentParser
from data.funk22 import get_e, get_well_position
from utils.enum import EType
from utils.file import save_file
from utils.logging import get_stdout_logger
from utils.plot import *
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


MERGE_COLS = ["sgRNA_0", "gene_symbol_0", "plate", "well", "tile", "cell_i", "cell_j"]
PC_COLS = [f"PC{i}" for i in range(64)]
TEST_RATIO = 0.4


def standardize(x_train, x_test):
    mu = x_train.mean(axis=0)
    sd = x_train.std(axis=0)
    x_train = (x_train - mu) / sd
    x_test = (x_test - mu) / sd
    return x_train, x_test


def get_embed(s3_bucket_name, train_embed_dpath, val_embed_dpath, test_embed_dpath, cellprofiler_fpath):
    s3 = boto3.client("s3")
    metadata_df, zc, zs = [], [], []
    for embed_dpath in (train_embed_dpath, val_embed_dpath, test_embed_dpath):
        response = s3.list_objects_v2(Bucket=s3_bucket_name, Prefix=embed_dpath)
        for obj in response["Contents"]:
            file_obj = io.BytesIO()
            s3.download_fileobj(s3_bucket_name, obj["Key"], file_obj)
            file_obj.seek(0)
            metadata_df_elem, zy_elem, ze_elem = pickle.load(file_obj)
            metadata_df.append(metadata_df_elem[MERGE_COLS])
            zc.append(zy_elem)
            zs.append(ze_elem)
    metadata_df = pd.concat(metadata_df)
    metadata_df.reset_index(drop=True, inplace=True)
    metadata_df.loc[:, "left_idx"] = metadata_df.index
    zc = np.concatenate(zc)
    zs = np.concatenate(zs)

    cellprofiler_df = pd.read_parquet(cellprofiler_fpath, engine="pyarrow")
    cellprofiler_df = cellprofiler_df[MERGE_COLS + PC_COLS]

    merged_df = metadata_df.merge(cellprofiler_df, how="inner", on=MERGE_COLS)
    cellprofiler = merged_df[PC_COLS].values
    zc = zc[merged_df.left_idx]
    zs = zs[merged_df.left_idx]
    metadata_df = merged_df[MERGE_COLS]
    return metadata_df, zc, zs, cellprofiler


def main(args):
    logger = get_stdout_logger()
    metadata_df, zc, zs, cellprofiler = get_embed(args.s3_bucket_name, args.train_embed_dpath, args.val_embed_dpath,
        args.test_embed_dpath, args.cellprofiler_fpath)
    metadata_df.loc[:, "Plate and well"] = get_e(metadata_df, EType.PLATE_WELL)
    metadata_df.loc[:, "Well position"] = get_well_position(metadata_df)
    target_names = ("Plate and well", "Well position")
    auroc, f1_score = {}, {}
    for embed_name in ("zc", "zs", "CellProfiler"):
        auroc[embed_name] = []
        f1_score[embed_name] = []
        if embed_name == "zc":
            x = zc
        elif embed_name == "zs":
            x = zs
        else:
            assert embed_name == "CellProfiler"
            x = cellprofiler
        for target_name in target_names:
            logger.log(logging.INFO, f"Fitting x={embed_name}, y={target_name}")
            y = metadata_df[target_name]
            valid_mask = ~y.isna()
            x_valid, y_valid = x[valid_mask], y[valid_mask]
            is_binary = len(y_valid.unique()) == 2
            if is_binary:
                y_valid = y_valid.astype("uint16")
                model = LogisticRegression()
                x_train, x_test, y_train, y_test = train_test_split(x_valid, y_valid, test_size=TEST_RATIO, random_state=0)
                x_train, x_test = standardize(x_train, x_test)
                model.fit(x_train, y_train)
                pred_class = model.predict(x_test)
                pred_prob = model.predict_proba(x_test)[:, 1]
                auroc_elem = metrics.roc_auc_score(y_test, pred_prob)
                f1_score_elem = metrics.f1_score(y_test, pred_class)
            else:
                model = LogisticRegression(multi_class="multinomial")
                x_train, x_test, y_train, y_test = train_test_split(x_valid, y_valid, test_size=TEST_RATIO, random_state=0)
                model.fit(x_train, y_train)
                pred_class = model.predict(x_test)
                pred_prob = model.predict_proba(x_test)
                auroc_elem = metrics.roc_auc_score(y_test, pred_prob, average="macro", multi_class="ovr")
                f1_score_elem = metrics.f1_score(y_test, pred_class, average="macro")
            auroc[embed_name].append(auroc_elem)
            f1_score[embed_name].append(f1_score_elem)
            logger.log(logging.INFO, f"AUROC={auroc_elem:.3f}, F1 score={f1_score_elem}")

    auroc = pd.DataFrame(auroc, index=target_names)
    f1_score = pd.DataFrame(f1_score, index=target_names)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    auroc.plot(ax=axes[0], kind="bar", legend=False)
    f1_score.plot(ax=axes[1], kind="bar", legend=False)
    for ax in axes:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
        ax.set_ylim(0, 1.1)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    axes[0].set_ylabel("AUROC")
    axes[1].set_ylabel("F1 score")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(handles), bbox_to_anchor=[0.5, 0])
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.275)

    os.makedirs(args.results_dpath, exist_ok=True)
    plt.savefig(os.path.join(args.results_dpath, "batch_effects.png"))
    save_file((auroc, f1_score), os.path.join(args.results_dpath, "batch_effects.pkl"))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--results_dpath", type=str, required=True)
    parser.add_argument("--s3_bucket_name", type=str, required=True)
    parser.add_argument("--train_embed_dpath", type=str, required=True)
    parser.add_argument("--val_embed_dpath", type=str, required=True)
    parser.add_argument("--test_embed_dpath", type=str, required=True)
    parser.add_argument("--cellprofiler_fpath", type=str, required=True)
    args = parser.parse_args()
    main(args)