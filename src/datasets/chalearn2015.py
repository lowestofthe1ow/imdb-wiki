"""
Make sure to run src/scripts/preprocess_chalearn.sh first.
Data from ChaLearn LAP 2015 should be in a single directory:

[chalearn_path directory]
├── raw
│   ├── test/
│   ├── train/
│   └── validation/
├── test/
├── train/
├── validation/
├── test.csv
├── train.csv
└── validation.csv

https://drive.google.com/drive/folders/1fe73NhI7OqmjnB-G6pr3xK8S-_XXgD6j?usp=sharing
"""

import cv2
import os
import pandas as pd
import torch
import torchvision.transforms as transforms

from PIL import Image
from pathlib import Path
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from tqdm import tqdm

from src.datasets.constants import TRAIN_TRANSFORMS, VAL_TEST_TRANSFORMS


class ChaLearn2015Dataset(Dataset):
    """Torch Dataset object for IMDB-WIKI"""

    def __init__(self, chalearn_path: str, train=True):
        """Initializes the dataset

        Args:
            imdbwiki_path:  Path to IMDB-WIKI data. Must contain /imdb_crop, /wiki_crop, and imdbwiki.csv
            path_prefix:    Path to prefix to the image paths in the CSV file
        """
        chalearn_train_df = pd.read_csv(Path(chalearn_path) / "train.csv")
        chalearn_valid_df = pd.read_csv(Path(chalearn_path) / "validation.csv")

        chalearn_train_df["file_name"] = chalearn_train_df["file_name"].apply(
            lambda path: os.path.join(chalearn_path, "train", path)
        )

        chalearn_valid_df["file_name"] = chalearn_valid_df["file_name"].apply(
            lambda path: os.path.join(chalearn_path, "validation", path)
        )

        self.labels = pd.concat(
            [
                chalearn_train_df,
                chalearn_valid_df,
            ]
        )
        self.labels["age"] = self.labels["apparent_age_avg"].round().astype("int")

        # Use different pre-transforms depending on if train or test/validation
        if train:
            self.transform = TRAIN_TRANSFORMS
        else:
            self.transform = VAL_TEST_TRANSFORMS

    def __len__(self):
        """Returns the size of the dataset"""
        return len(self.labels)

    def __getitem__(self, index):
        try:
            file_name = self.labels.iloc[index]["file_name"]
            image = Image.open(file_name).convert("RGB")
            label = self.labels.iloc[index]["age"]
            image = self.transform(image)

        except Exception as e:
            print(f"Error opening `{file_name}`:\n{e}")

        return image, label


def main():
    print("Sanity-checking ChaLearn LAP 2015 dataset...")
    dataset = ChaLearn2015Dataset("data/chalearn_2015")
    for i in tqdm(range(dataset.__len__())):
        dataset.__getitem__(i)


if __name__ == "__main__":
    main()
