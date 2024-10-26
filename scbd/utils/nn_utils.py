import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Sampler
from utils.const import UINT32_MAX


class HParams(dict):
    def __getattr__(self, key):
        return self[key]

    def __setattr__(self, key, value):
        self[key] = value


class DataFrameDataset(Dataset):
    def __init__(self, df, img_pixels):
        self.df = df
        self.transforms = T.Compose([
            T.Resize((img_pixels, img_pixels)),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        x = self.transforms(Image.open(self.df.fpath.iloc[idx]).convert("RGB"))
        return x, self.df.iloc[idx]


class YBatchSampler(Sampler):
    def __init__(self, y, pmf_y, batch_size, y_per_batch):
        super().__init__()
        assert batch_size % y_per_batch == 0
        self.y = y
        self.pmf_y = pmf_y
        self.y_per_batch = y_per_batch
        self.batch_size_per_y = batch_size // y_per_batch
        self.y_unique = np.unique(y)
        self.y_to_idxs = {y_value.item(): np.where(y == y_value)[0] for y_value in self.y_unique}

    def __iter__(self):
        while True:
            y_batch = np.random.choice(self.y_unique, size=self.y_per_batch, replace=False, p=self.pmf_y)
            idxs = []
            for y_value in y_batch:
                y_idxs = self.y_to_idxs[y_value]
                if self.batch_size_per_y < len(y_idxs):
                    idxs += np.random.choice(y_idxs, size=self.batch_size_per_y, replace=False).tolist()
                else:
                    idxs += y_idxs.tolist()
            yield idxs

    def __len__(self):
        return UINT32_MAX


def collate(batch):
    x, df = zip(*batch)
    x = torch.stack(x, dim=0)
    df = pd.concat(df, axis=1).T
    df.y = df.y.astype("int64")
    if df.e.isna().any():
        df.e = df.e.astype("float32")
    else:
        df.e = df.e.astype("int64")
    return x, df


def get_dataloader(dataset, batch_size, workers, shuffle, drop_last):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers, collate_fn=collate,
        pin_memory=True, drop_last=drop_last, persistent_workers=True)


def shuffle_batch(x):
    batch_size = len(x)
    idxs = torch.randperm(batch_size)
    return x[idxs]