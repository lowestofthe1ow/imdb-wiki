"""
Data from APPA-REAL should be in a single directory:

[appareal_path directory]
├── allcategories_test.csv
├── allcategories_train.csv
├── allcategories_valid.csv
├── gt_avg_test.csv
├── gt_avg_train.csv
├── gt_avg_valid.csv
├── gt_test.csv
├── gt_train.csv
├── gt_valid.csv
├── test/
├── train/
└── valid/

https://data.chalearnlap.cvc.uab.cat/AppaRealAge/appa-real-release.zip
"""

import pandas as pd
import torch
import torchvision.transforms as transforms

from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm

from src.datasets.constants import TRAIN_TRANSFORMS, VAL_TEST_TRANSFORMS


class APPAREALDataset(Dataset):
    """Torch Dataset for APPA REAL"""

    def __init__(self, appareal_path: str, train=True):
        """
        Args:
            csv_path:   Path to the directory containing the APPA-REAL image metadata
                        (contains gt_avg_train.csv, gt_avg_test.csv, gt_avg_valid.csv)
        """
        self.appareal_path = appareal_path

        appareal_train_df = pd.read_csv(Path(appareal_path) / "gt_avg_train.csv")
        appareal_test_df = pd.read_csv(Path(appareal_path) / "gt_avg_test.csv")
        appareal_valid_df = pd.read_csv(Path(appareal_path) / "gt_avg_valid.csv")

        # ..._face.jpg is the cropped face image
        appareal_train_df["file_name"] = (
            Path(appareal_path)
            / "train"
            / (appareal_train_df["file_name"] + "_face.jpg")
        )
        """
        appareal_test_df["file_name"] = (
            Path(appareal_path) / "test" / (appareal_test_df["file_name"] + "_face.jpg")
        )
        """
        appareal_valid_df["file_name"] = (
            Path(appareal_path)
            / "valid"
            / (appareal_valid_df["file_name"] + "_face.jpg")
        )

        self.labels = pd.concat(
            [
                appareal_train_df,
                # appareal_test_df, # Don't include the test data since it will be used for MAE calculation
                appareal_valid_df,
            ]
        )
        self.labels["age"] = self.labels["apparent_age_avg"].round().astype("int")

        if train:
            self.transform = TRAIN_TRANSFORMS
        else:
            self.transform = VAL_TEST_TRANSFORMS

    def __len__(self):
        """"""
        return len(self.labels)

    def __getitem__(self, index) -> tuple[Image.Image, int]:
        """
        Returns tensor representation of the Image and apparent age given index
        ! We assume the folder to the images is in the same folder as the csv_path provided
        """

        try:
            file_name = self.labels.iloc[index]["file_name"]
            image = Image.open(file_name).convert("RGB")
            label = self.labels.iloc[index]["age"]
            image = self.transform(image)
        except Exception as e:
            print(f"Error opening `{file_name}`:\n{e}")

        return image, label


def main():
    print("Sanity-checking APPA-REAL dataset...")
    dataset = APPAREALDataset("data/appa-real-release")
    for i in tqdm(range(dataset.__len__())):
        dataset.__getitem__(i)


if __name__ == "__main__":
    main()
