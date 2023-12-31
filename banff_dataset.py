import os
import typing

import numpy as np
import h5py
import torch
from torch import Tensor
from torch.utils.data import Dataset
import pandas as pd

ATTRIBUTES = ["bx_tma", "bx_atn", "bx_g", "bx_cg", "bx_mm", "bx_i",
              "bx_t", "bx_ti", "bx_ifta", "bx_iifta", "bx_tifta",
              "bx_ptc", "bx_v", "bx_cv", "bx_ah"]
ORDINAL_ATTRIBUTES = ["bx_g", "bx_cg", "bx_mm", "bx_i", "bx_t", "bx_iifta", "bx_tifta",
                      "bx_ptc", "bx_v", "bx_cv", "bx_ah"]
ATTRIBUTES_WITH_MISSING = ["bx_px_v", "bx_px_cv"]
BINARY_ATTRIBUTES = ["bx_tma", "bx_atn"]
CONTINUOUS_ATTRIBUTES = ["bx_ti", "bx_ifta"]


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


def transform_scores(scores: pd.Series, device: torch.device) -> typing.List:
    """
    Apply the following transformations: perform the one-hot encoding of the ordinal attributes, making sure to
    add a new value (-1) for attributes which may present missing values; rescale the continuous attributes to [0,1];
    leave the binary attributes as they are.
    :param scores: the Banff scores to transform (Pandas Dataframe)
    :param device: the device on which to store the transformed scores
    :return: the transformed Banff scores (Numpy array)
    """
    labels = []
    # Iterate through all the attributes
    for attribute in ATTRIBUTES:
        if attribute in BINARY_ATTRIBUTES:
            labels.append(int(scores[attribute]))
        elif attribute in CONTINUOUS_ATTRIBUTES:
            labels.append(scores[attribute] / 100)
        elif attribute in ORDINAL_ATTRIBUTES:
            # If the attribute is ordinal, we need to perform the one-hot encoding
            # First, we need to check if the attribute can have missing values
            if attribute in ATTRIBUTES_WITH_MISSING:
                # If the attribute has missing values, we need to add a new value to the one-hot encoding
                # This new value will be -1
                if scores[attribute] is None:
                    scores[attribute] = -1

                labels.append(torch.eye(5, dtype=torch.long, device=device)[int(scores[attribute]) + 1])
            else:
                labels.append(torch.eye(4, dtype=torch.long, device=device)[int(scores[attribute])])
        else:
            raise ValueError(f"Unknown attribute kind {attribute}")

    return labels


def collate(batch: typing.List) -> typing.Tuple[Tensor, Tensor, typing.List[Tensor]]:
    """
    Collate function for the Banff dataset. This function is used by the PyTorch DataLoader.
    :param batch: A list of tuples containing the features, coordinates and labels.
    :return: A tuple containing the features, coordinates and labels of the batch all together as tensors.
    """
    features = torch.cat([item[0] for item in batch], dim=0)
    coords = torch.cat([item[1] for item in batch], dim=0)

    # The labels are lists of tensors, so we need to concatenate them separately
    labels = []
    for i in range(len(batch[0][2])):
        labels.append(torch.cat([item[2][i] for item in batch], dim=0))

    return features, coords, labels


def get_feature_bag_path(data_dir: str, slide_id: str) -> str:
    return os.path.join(data_dir, f"{slide_id}.h5")


class BanffDataset(Dataset):
    """
    Dataset class for the Banff lesion scores.
    """

    def __init__(self, data_dir: str, banff_scores_csv_filepath: str, device: torch.device):
        """
        :param data_dir: The directory containing the feature bags, i.e. the h5 files of the slides.
        :param banff_scores_csv_filepath: The path to the csv file containing the Banff scores.
        """
        self.banff = pd.read_csv(banff_scores_csv_filepath, delimiter=",")
        self.banff.set_index("PATIENT_ID", inplace=True)
        self.data_dir = data_dir
        self.device = device

    def __getitem__(self, idx) -> typing.Tuple[Tensor, Tensor, Tensor]:
        patient_id = self.banff.index[idx]
        scores = self.banff.loc[patient_id, ATTRIBUTES]
        scores = transform_scores(scores, self.device)

        full_path = get_feature_bag_path(self.data_dir, patient_id)
        with h5py.File(full_path, "r") as hdf5_file:
            features = hdf5_file["features"][:]
            coords = hdf5_file["coords"][:]

        features = torch.from_numpy(features)
        scores = torch.from_numpy(scores)

        # Load coordinates as float32 and normalize them to be in the range [0, 1]
        coords = torch.from_numpy(coords).float()
        coords[:, 0] = (coords[:, 0] - coords[:, 0].min()) / (coords[:, 0].max() - coords[:, 0].min())
        coords[:, 1] = (coords[:, 1] - coords[:, 1].min()) / (coords[:, 1].max() - coords[:, 1].min())

        return features, coords, scores

    def __len__(self) -> int:
        return len(self.banff)
