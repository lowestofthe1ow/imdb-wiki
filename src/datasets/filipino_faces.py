"""
Data from Filipino Faces should be in a single directory:

[filipino_faces_path directory]
├── labels.csv
└── faces/

The CSV should contain: path, birth_year, date_taken, age, gender
"""

import os
import pandas as pd
import torch
import torchvision.transforms as transforms

from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset
from tqdm import tqdm

from src.datasets.constants import TRAIN_TRANSFORMS, VAL_TEST_TRANSFORMS


class FilipinoFacesDataset(Dataset):
    """Torch Dataset for Filipino Faces"""

    def __init__(self, filipino_faces_path: str, train=True):
        """
        Args:
            filipino_faces_path:    Path to the directory containing the Filipino faces data
                                    (contains labels.csv and faces/ subdirectory)
            train:                  Whether to use training or validation/test transforms
        """
        self.labels = pd.read_csv(Path(filipino_faces_path) / "labels.csv")

        self.labels["full_path"] = self.labels["path"].apply(
            lambda path: os.path.join(filipino_faces_path, path)
        )

        if train:
            self.transform = TRAIN_TRANSFORMS
        else:
            self.transform = VAL_TEST_TRANSFORMS

    def __len__(self):
        """Returns the size of the dataset"""
        return len(self.labels)

    def __getitem__(self, index) -> tuple[Image.Image, int]:
        """
        Returns tensor representation of the image and age given index
        """
        try:
            file_name = self.labels.iloc[index]["full_path"]
            image = Image.open(file_name).convert("RGB")
            label = self.labels.iloc[index]["age"]
            image = self.transform(image)
        except Exception as e:
            print(f"Error opening `{file_name}`:\n{e}")

        return image, label


def main():
    print("Sanity-checking Filipino Faces dataset...")
    dataset = FilipinoFacesDataset("data/filipino_faces_dataset")
    for i in tqdm(range(dataset.__len__())):
        dataset.__getitem__(i)


if __name__ == "__main__":
    main()
