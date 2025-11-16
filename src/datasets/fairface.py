import numpy as np
import os
import pandas as pd
import torch
import torchvision.transforms as transforms

from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset

from src.datasets.constants import TRAIN_TRANSFORMS, VAL_TEST_TRANSFORMS


class FairFaceDataset(Dataset):
    """Torch Dataset for FairFace"""

    def __init__(self, fairface_path: str, train=True):
        """
        Args:
            fairface_path: Path that contains
                `fairface_label_train.csv`, `fairface_label_val.csv`,
                and `fairface-img-margin025-trainval/`.
        """
        fairface_train_df = pd.read_csv(
            Path(fairface_path) / "fairface_label_train.csv"
        )
        fairface_val_df = pd.read_csv(Path(fairface_path) / "fairface_label_val.csv")
        fairface_df = pd.concat([fairface_train_df, fairface_val_df])

        def mean_from_age_range_string(age_range_str: str):
            """String '1-10' will return 5.5"""
            if "-" in age_range_str:
                first, second = age_range_str.split("-")
                return (int(first) + int(second)) / 2
            elif age_range_str == "more than 70":  # special case
                return 70
            else:
                return int(age_range_str)

        self.labels = pd.DataFrame(
            {
                "full_path": fairface_df["file"].apply(
                    lambda file_path: os.path.join(
                        fairface_path, "fairface-img-margin025-trainval", file_path
                    )
                ),
                "age": fairface_df["age"].apply(
                    lambda age_range_str: mean_from_age_range_string(age_range_str)
                ),
                "race": fairface_df["race"],
                "gender": fairface_df["gender"],
            }
        )

        if train:
            self.transform = TRAIN_TRANSFORMS
        else:
            self.transform = VAL_TEST_TRANSFORMS

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index) -> tuple[Image.Image, int]:
        try:
            image_path = self.labels.iloc[index]["full_path"]
            image = Image.open(image_path).convert("RGB")
            image = self.transform(image)
            label = self.labels.iloc[index]["age"]
        except Exception as e:
            print(f"Error opening `{file_name}`:\n{e}")

        return image, label


def main():
    print(FairFaceDataset("data/fairface")[0])


if __name__ == "__main__":
    main()
