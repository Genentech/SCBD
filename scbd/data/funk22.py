import lmdb
import numpy as np
import os
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as T
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from utils.enum import ExperimentGroup, EType
from utils.nn_utils import YBatchSampler, collate


IMG_CHANNEL_NAMES = [
    "DNA damage",
    "F-actin",
    "DNA content",
    "Microtubules"
]
PLATES = [
    "20200202_6W-LaC024A",
    "20200202_6W-LaC024D",
    "20200202_6W-LaC024E",
    "20200202_6W-LaC024F",
    "20200206_6W-LaC025A",
    "20200206_6W-LaC025B"
]
PH_DIMS = (2960, 2960)
SPLIT_RATIOS = [0.83, 0.02, 0.15]
IMG_ORIGINAL_PIXELS = 100
IMG_PIXELS = 32
IMG_CHANNELS = len(IMG_CHANNEL_NAMES)
PLOT_SINGLE_CHANNEL = True


class Arcsinh(nn.Module):
    def forward(self, x):
        return torch.arcsinh(x)


class Funk22Dataset(Dataset):
    def __init__(self, data_dpath, df):
        self.df = df
        self.lmdb_treatment = lmdb.Environment(get_dpath_treatment(data_dpath), readonly=True, readahead=False, lock=False)
        self.lmdb_control = lmdb.Environment(get_dpath_control(data_dpath), readonly=True, readahead=False, lock=False)
        self.transforms = T.Compose([
            T.Resize((IMG_PIXELS, IMG_PIXELS)),
            Arcsinh(),
            T.Normalize(7., 7.)
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        is_treatment = self.df.group.iloc[idx] == ExperimentGroup.TREATMENT
        lmdb = self.lmdb_treatment if is_treatment else self.lmdb_control
        img_name = f'{row.UID}_{row.plate}_{row.well}_{row.tile}_{row.gene_symbol_0}_{row["index"]}'
        with lmdb.begin(write=False, buffers=True) as txn:
            buf = txn.get(img_name.encode())
            x = np.frombuffer(buf, dtype="uint16")
        x = x.reshape((IMG_CHANNELS, IMG_ORIGINAL_PIXELS, IMG_ORIGINAL_PIXELS))
        x = torch.tensor(x)
        x = self.transforms(x)
        return x, row


def get_dpath_treatment(data_dpath):
    return os.path.join(data_dpath, "funk22_lmdb_shuffled", "perturbed")


def get_dpath_control(data_dpath):
    return os.path.join(data_dpath, "funk22_lmdb_shuffled", "ntc")


def get_tile_grid(shape):
    if shape == "6W_ph":
        rows = [7, 13, 17, 21, 25, 27, 29, 31, 33, 33, 35, 35, 37, 37, 39, 39, 39, 41, 41, 41, 41, 41, 41, 41, 39, 39,
            39, 37, 37, 35, 35, 33, 33, 31, 29, 27, 25, 21, 17, 13, 7]
    elif shape == "6W_sbs":
        rows = [5, 9, 13, 15, 17, 17, 19, 19, 21, 21, 21, 21, 21, 19, 19, 17, 17, 15, 13, 9, 5]
    elif isinstance(shape, list):
        rows = shape
    else:
        raise ValueError
    c, r = len(rows), max(rows)
    tile_grid = np.full((r, c), np.nan)

    next_site = 0
    for col, row_sites in enumerate(rows):
        start = int((r - row_sites) / 2)
        if col % 2 == 0:
            tile_grid[start:start + row_sites, col] = range(next_site, next_site + row_sites)
        else:
            tile_grid[start:start + row_sites, col] = range(next_site, next_site + row_sites)[::-1]
        next_site += row_sites
    return tile_grid


def get_center_coords(disk, center, n_points):
    distances = np.sqrt((np.indices(disk.shape).T - center) ** 2).sum(-1)
    mask = np.isnan(disk)
    masked_distances = distances[~mask]
    sorted_distances = np.sort(masked_distances)
    radius = sorted_distances[n_points]
    # Circular mask
    y, x = np.ogrid[-center[0]:disk.shape[0] - center[0], -center[1]:disk.shape[1] - center[1]]
    mask = x * x + y * y <= radius * radius
    # Get the points within the circular mask and inside the disk
    center_coordinates = np.column_stack(np.where(np.logical_and(mask, ~np.isnan(disk))))
    return center_coordinates


def find_tile_adjacent_to_nan(tile_grid):
    center = np.array(tile_grid.shape) // 2
    edge_coords = []
    kernel = np.array([[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0], [1, 1]])
    for i in range(tile_grid.shape[0]):
        for j in range(tile_grid.shape[1]):
            if not np.isnan(tile_grid[i, j]):
                # If the point is on the edge of the array
                if i == 0 or i == tile_grid.shape[0] - 1 or j == 0 or j == tile_grid.shape[1] - 1:
                    edge_coords.append([i, j])
                else:
                    # Check all neighbors
                    for k in range(8):
                        ni, nj = np.array([i, j]) + kernel[k]
                        # If the neighbor is nan, this point is at the edge
                        if np.isnan(tile_grid[ni, nj]):
                            edge_coords.append([i, j])
                            break
    # Select a similar number of coordinates from the center
    center_coords = np.array(get_center_coords(tile_grid, center, len(edge_coords)))
    edge_coords = np.array(edge_coords)
    center_tiles = tile_grid[center_coords[:, 0], center_coords[:, 1]].astype("int").astype("str")
    edge_tiles = tile_grid[edge_coords[:, 0], edge_coords[:, 1]].astype("int").astype("str")
    return edge_tiles, center_tiles


def get_y_name_to_idx(df):
    y_names_unique = sorted(df.gene_symbol_0.unique())
    y_name_to_idx = {y_name: i for i, y_name in enumerate(y_names_unique)}
    return y_name_to_idx


def get_y(df):
    y_name_to_idx = get_y_name_to_idx(df)
    y = df.gene_symbol_0.map(y_name_to_idx)
    return y


def get_e(df, e_type):
    if e_type == EType.NONE:
        return 0
    elif e_type == EType.PLATE_WELL:
        e_names = df["plate"] + df["well"]
    else:
        assert e_type == EType.PLATE_WELL_TILE
        well_position = get_well_position(df)
        e_names = df["plate"] + df["well"] + well_position.astype(str)
    e_names_unique = sorted(e_names.unique())
    e_name_to_idx = {batch_name: i for i, batch_name in enumerate(e_names_unique)}
    e = e_names.map(e_name_to_idx)
    return e


def get_well_position(df):
    tile_grid = get_tile_grid("6W_ph")
    edge_tiles, center_tiles = find_tile_adjacent_to_nan(tile_grid)
    well_position = pd.Series([np.nan] * len(df))
    well_position[df.tile.astype(str).isin(edge_tiles)] = 0
    well_position[df.tile.astype(str).isin(center_tiles)] = 1
    return well_position


def get_fov_position(df):
    fov_position = pd.Series([np.nan] * len(df))
    distance_from_center = np.abs(df[["cell_i", "cell_j"]] - df[["cell_i", "cell_j"]].mean(axis=0)).max(axis=1)
    fov_position[distance_from_center >= 1200] = 0
    fov_position[distance_from_center < 450] = 1
    return df


def get_group_df(dpath, group):
    df = pd.read_csv(os.path.join(dpath, "key.csv"), dtype={"UID": str})
    df = df[df.plate.isin(PLATES)]
    radius = IMG_ORIGINAL_PIXELS / 2
    df = df[
        df.cell_i.between(radius, PH_DIMS[0] - radius) &
        df.cell_j.between(radius, PH_DIMS[1] - radius)
    ]
    df["group"] = group
    return df


def get_df(data_dpath, e_type):
    df_treatment = get_group_df(get_dpath_treatment(data_dpath), ExperimentGroup.TREATMENT)
    df_control = get_group_df(get_dpath_control(data_dpath), ExperimentGroup.CONTROL)
    df = pd.concat((df_treatment, df_control))
    df.reset_index(drop=True, inplace=True)
    df["y"] = get_y(df)
    df["e"] = get_e(df, e_type)
    df = df.sample(frac=1)
    return df


def split_df(df):
    _, val_ratio, test_ratio = SPLIT_RATIOS
    df_train, df_remainder = train_test_split(df, test_size=val_ratio + test_ratio)
    df_val, df_test = train_test_split(df_remainder, test_size=test_ratio / (val_ratio + test_ratio))
    return df_train, df_val, df_test


def get_dataloader(dataset, data_split, pmf_y, batch_size, y_per_batch, workers):
    is_train = data_split is None
    if is_train:
        batch_sampler = YBatchSampler(dataset.df.y.values, pmf_y, batch_size, y_per_batch)
        return DataLoader(dataset, batch_sampler=batch_sampler, num_workers=workers, collate_fn=collate, pin_memory=True,
            persistent_workers=True)
    else:
        return DataLoader(dataset, batch_size=batch_size, num_workers=workers, collate_fn=collate, pin_memory=True,
            persistent_workers=True)


def get_data(hparams):
    df = get_df(hparams.data_dpath, hparams.e_type)
    df_train, df_val, df_test = split_df(df)

    dataset_train = Funk22Dataset(hparams.data_dpath, df_train)
    dataset_val = Funk22Dataset(hparams.data_dpath, df_val)
    dataset_test = Funk22Dataset(hparams.data_dpath, df_test)

    y_train = df_train.y.values
    pmf_y = np.bincount(y_train)
    pmf_y = pmf_y / pmf_y.sum()

    data_train = get_dataloader(dataset_train, hparams.data_split, pmf_y, hparams.batch_size, hparams.y_per_batch, hparams.workers)
    data_val = get_dataloader(dataset_val, hparams.data_split, pmf_y, hparams.batch_size, hparams.y_per_batch, hparams.workers)
    data_test = get_dataloader(dataset_test, hparams.data_split, pmf_y, hparams.batch_size, hparams.y_per_batch, hparams.workers)

    metadata = {
        "img_channels": IMG_CHANNELS,
        "y_size": df.y.max() + 1,
        "e_size": df.e.max() + 1
    }
    return data_train, data_val, data_test, metadata