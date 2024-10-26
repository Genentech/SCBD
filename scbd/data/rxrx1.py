import numpy as np
import os
import pandas as pd
import wilds.datasets.rxrx1_dataset
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from utils.nn_utils import DataFrameDataset, YBatchSampler, collate


IMG_PIXELS = 32
IMG_CHANNELS = 3
Y_SIZE = 1139
E_SIZE = 33
TRAIN_RATIO = 0.9


def get_dataloader(dataset, is_trainval, pmf_y, batch_size, y_per_batch, workers):
    if is_trainval:
        batch_sampler = YBatchSampler(dataset.df.y.values, pmf_y, batch_size, y_per_batch)
        return DataLoader(dataset, batch_sampler=batch_sampler, num_workers=workers, collate_fn=collate, pin_memory=True,
            persistent_workers=True)
    else:
        return DataLoader(dataset, batch_size=batch_size, num_workers=workers, collate_fn=collate, pin_memory=True,
            persistent_workers=True)


def split_train_val(df_trainval):
    y = df_trainval.y
    y_unique = np.unique(y)
    y_to_idxs = {y_value.item(): np.where(y == y_value)[0] for y_value in y_unique}
    train_idxs, val_idxs = [], []
    for idxs in y_to_idxs.values():
        train_idxs_y, val_idxs_y = train_test_split(idxs, test_size=1. - TRAIN_RATIO, random_state=0)
        train_idxs.extend(train_idxs_y)
        val_idxs.extend(val_idxs_y)
    return df_trainval.iloc[train_idxs], df_trainval.iloc[val_idxs]


def get_data(hparams):
    wilds_dataset = wilds.datasets.rxrx1_dataset.RxRx1Dataset(root_dir=hparams.data_dpath, download=True)
    df = {}
    df["fpath"] = np.array([os.path.join(hparams.data_dpath,  "rxrx1_v1.0", fpath) for fpath in wilds_dataset._input_array])
    df["y"] = wilds_dataset._metadata_array[:, wilds_dataset._metadata_fields.index("y")]
    df["e"] = wilds_dataset._metadata_array[:, wilds_dataset._metadata_fields.index("experiment")]
    df = pd.DataFrame(df)

    trainval_idxs = np.where(wilds_dataset._split_array == wilds_dataset._split_dict["train"])[0]
    test_idxs = np.where(wilds_dataset._split_array == wilds_dataset._split_dict["test"])[0]

    df_trainval = df.iloc[trainval_idxs]
    df_train, df_val = split_train_val(df_trainval)
    df_test = df.iloc[test_idxs]

    train_envs_nonconsecutive = sorted(df_train.e.unique().tolist())
    train_envs_consecutive = np.argsort(train_envs_nonconsecutive)
    nonconsecutive_to_consecutive = dict((u, o) for u, o in zip(train_envs_nonconsecutive, train_envs_consecutive))
    df_train.loc[:, "e"] = [nonconsecutive_to_consecutive[e] for e in df_train.e]
    df_val.loc[:, "e"] = [nonconsecutive_to_consecutive[e] for e in df_val.e]
    df_test.loc[:, "e"] = np.nan

    df_train = df_train.sample(frac=1.).reset_index(drop=True)
    df_val = df_val.sample(frac=1.).reset_index(drop=True)
    df_test = df_test.sample(frac=1.).reset_index(drop=True)

    y_train = df_train.y.values
    pmf_y = np.bincount(y_train)
    pmf_y = pmf_y / pmf_y.sum()

    data_train = get_dataloader(DataFrameDataset(df_train, IMG_PIXELS), True, pmf_y, hparams.batch_size, hparams.y_per_batch,
        hparams.workers)
    data_val = get_dataloader(DataFrameDataset(df_val, IMG_PIXELS), True, pmf_y, hparams.batch_size, hparams.y_per_batch,
        hparams.workers)
    data_test = get_dataloader(DataFrameDataset(df_test, IMG_PIXELS), False, pmf_y, hparams.batch_size, hparams.y_per_batch,
        hparams.workers)

    metadata = {
        "img_channels": IMG_CHANNELS,
        "y_size": Y_SIZE,
        "e_size": E_SIZE
    }
    return data_train, data_val, data_test, metadata