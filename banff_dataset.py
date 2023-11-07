import os
import typing
from typing import Tuple

import h5py
import torch
from torch import Tensor
from torch.utils.data import Dataset
import pandas as pd

ATTRIBUTES = ["bx_tma", "bx_atn", "bx_g", "bx_cg", "bx_mm", "bx_i",
              "bx_t", "bx_ti", "bx_ifta", "bx_iifta", "bx_tifta",
              "bx_ptc", "bx_v", "bx_cv", "bx_ah"]


# |-------------------------------------------------------------------|
# |               Description of the attributes                       |
# |-------------------------------------------------------------------|
# |  bx_px_tma     | Thrombotic microangiopathy                       |
# |  bx_px_atn     | Acute tubular injury                             |
# |  bx_px_g       | Glomerulitis                                     |
# |  bx_px_cg      | Transplant glomerulopathy                        |
# |  bx_px_mm      | Mesangial increase                               |
# |  bx_px_i       | Interstitial inflammation                        |
# |  bx_px_t       | Tubulitis                                        |
# |  bx_px_ti      | Total inflammation                               |
# |  bx_px_ifta    | Interstitial fibrosis and tubular atrophy (IFTA) |
# |  bx_px_iifta   | Interstitial inflammation in IFTA                |
# |  bx_px_tifta   | Tubulitis in IFTA                                |
# |  bx_px_ptc     | Peritubular capillaritis                         |
# |  bx_px_v       | Endothelialitis/arteriitis                       |
# |  bx_px_cv      | Intimal fibrous thickening                       |
# |  bx_px_ah      | Arteriolar hyalinosis                            |
# |-------------------------------------------------------------------|


def collate(batch: typing.List) -> tuple[Tensor, Tensor, Tensor]:
    """
    Collate function for the Banff dataset. This function is used by the PyTorch DataLoader.
    :param batch: A list of tuples containing the features, coordinates and labels.
    :return: A tuple containing the features, coordinates and labels of the batch all together as tensors.
    """
    features = torch.cat([item[0] for item in batch], dim=0)
    coords = torch.cat([item[1] for item in batch], dim=0)
    labels = torch.cat([item[2] for item in batch], dim=0)

    return features, coords, labels


def get_feature_bag_path(data_dir: str, slide_id: str) -> str:
    return os.path.join(data_dir, f"{slide_id}.h5")


class BanffDataset(Dataset):
    """
    Dataset class for the Banff lesion scores.
    """
    def __init__(self, data_dir: str, banff_scores_csv_filepath: str):
        """
        :param data_dir: The directory containing the feature bags, i.e. the h5 files of the slides.
        :param banff_scores_csv_filepath: The path to the csv file containing the Banff scores.
        """
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
