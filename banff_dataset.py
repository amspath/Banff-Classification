import os
import typing
from typing import Tuple

import h5py
import torch
from torch import Tensor
from torch.utils.data import Dataset
import pandas as pd

ATTRIBUTES = ["bx_p1_tma", "bx_p1_atn", "bx_p1_g", "bx_p1_cg", "bx_p1_mm", "bx_p1_i",
              "bx_p1_t", "bx_p1_ti", "bx_p1_ifta", "bx_p1_iifta", "bx_p1_tifta",
              "bx_p1_ptc", "bx_p1_v", "bx_p1_cv", "bx_p1_ah"]


def collate(batch: typing.List) -> tuple[Tensor, Tensor, Tensor]:
    features = torch.cat([item[0] for item in batch], dim=0)
    coords = torch.cat([item[1] for item in batch], dim=0)
    labels = torch.cat([item[2] for item in batch], dim=0)

    return features, coords, labels


def get_feature_bag_path(data_dir: str, slide_id: str) -> str:
    return os.path.join(data_dir, f"{slide_id}.h5")


class BanffDataset(Dataset):
    def __init__(self, data_dir: str, banff_scores_csv_filepath: str):
        self.banff = pd.read_csv(banff_scores_csv_filepath, delimiter=",")
        self.banff.set_index("PATIENT_ID", inplace=True)
        self.data_dir = data_dir

    def __getitem__(self, idx) -> Tuple[Tensor, Tensor, Tensor]:
        patient_id = self.banff.index[idx]
        scores = self.banff.loc[patient_id, ATTRIBUTES].values

        full_path = get_feature_bag_path(self.data_dir, patient_id)
        with h5py.File(full_path, "r") as hdf5_file:
            features = hdf5_file["features"][:]
            coords = hdf5_file["coords"][:]

        features = torch.from_numpy(features)
        scores = torch.from_numpy(scores)
        coords = torch.from_numpy(coords)

        return features, coords, scores

    def __len__(self) -> int:
        return len(self.banff)
