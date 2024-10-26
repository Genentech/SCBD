import numpy as np
import os
import pandas as pd
import wilds.datasets.camelyon17_dataset
from utils.nn_utils import DataFrameDataset, get_dataloader


IMG_PIXELS = 32
IMG_CHANNELS = 3
Y_SIZE = 2
E_SIZE = 3


def get_data(hparams):
    wilds_dataset = wilds.datasets.camelyon17_dataset.Camelyon17Dataset(root_dir=hparams.data_dpath, download=True)
    df = {}
    df["fpath"] = np.array([os.path.join(hparams.data_dpath,  "camelyon17_v1.0", fpath) for fpath in wilds_dataset._input_array])
    df["y"] = wilds_dataset._metadata_array[:, wilds_dataset._metadata_fields.index("y")]
    df["e"] = wilds_dataset._metadata_array[:, wilds_dataset._metadata_fields.index("hospital")]
    df = pd.DataFrame(df)

    train_idxs = np.where(wilds_dataset._split_array == wilds_dataset._split_dict["train"])[0]
    val_idxs = np.where(wilds_dataset._split_array == wilds_dataset._split_dict["id_val"])[0]
    test_idxs = np.where(wilds_dataset._split_array == wilds_dataset._split_dict["test"])[0]

    df_train = df.iloc[train_idxs]
    df_val = df.iloc[val_idxs]
    df_test = df.iloc[test_idxs]

    train_envs_nonconsecutive = sorted(df_train.e.unique().tolist())
    train_envs_consecutive = np.argsort(train_envs_nonconsecutive)
    nonconsecutive_to_consecutive = dict((u, o) for u, o in zip(train_envs_nonconsecutive, train_envs_consecutive))
    df_train.loc[:, "e"] = [nonconsecutive_to_consecutive[e] for e in df_train.e]
    df_val.loc[:, "e"] = [nonconsecutive_to_consecutive[e] for e in df_val.e]
    df_test.loc[:, "e"] = df_test.e.astype(float)
    df_test.loc[:, "e"] = np.nan

    df_train = df_train.sample(frac=1.).reset_index(drop=True)
    df_val = df_val.sample(frac=1.).reset_index(drop=True)
    df_test = df_test.sample(frac=1.).reset_index(drop=True)

    data_train = get_dataloader(DataFrameDataset(df_train, IMG_PIXELS), hparams.batch_size, hparams.workers, True, False)
    data_val_id = get_dataloader(DataFrameDataset(df_val, IMG_PIXELS), hparams.batch_size, hparams.workers, False, True)
    data_test = get_dataloader(DataFrameDataset(df_test, IMG_PIXELS), hparams.batch_size, hparams.workers, False, False)

    metadata = {
        "img_channels": IMG_CHANNELS,
        "y_size": Y_SIZE,
        "e_size": E_SIZE
    }
    return data_train, data_val_id, data_test, metadata