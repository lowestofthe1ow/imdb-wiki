import cv2
import os
import pandas as pd
import torch
import torchvision.transforms as transforms

from PIL import Image
from pathlib import Path
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

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
        chalearn_valid_df = pd.read_csv(Path(chalearn_path) / "valid.csv")

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
    transform = transforms.ToPILImage()
    image, label = IMDBWIKIDataset("data/chalearn_path/").__getitem__(0)
    transform(image).show()


if __name__ == "__main__":
    main()
