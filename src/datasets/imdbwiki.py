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


class IMDBWIKIDataset(Dataset):
    """Torch Dataset object for IMDB-WIKI"""

    def __init__(self, imdbwiki_path: str, train=True):
        """Initializes the dataset

        Args:
            imdbwiki_path:  Path to IMDB-WIKI data. Must contain /imdb_crop, /wiki_crop, and imdbwiki.csv
            path_prefix:    Path to prefix to the image paths in the CSV file
        """
        self.labels = pd.read_csv(Path(imdbwiki_path) / "imdbwiki.csv")

        self.labels["full_path"] = self.labels["full_path"].apply(
            lambda path: os.path.join(imdbwiki_path, path)
        )

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
            image_path = self.labels.iloc[index]["full_path"]
            image = Image.open(image_path).convert("RGB")
            label = self.labels.iloc[index]["age"]
            image = self.transform(image)
        except Exception as e:
            print(f"Error opening `{file_name}`:\n{e}")

        return image, label


def main():
    transform = transforms.ToPILImage()
    image, label = IMDBWIKIDataset("data/imdb-wiki/").__getitem__(0)
    transform(image).show()


if __name__ == "__main__":
    main()
