from pathlib import Path
import pandas as pd
import torch
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import os

# Known values for the ImageNet dataset mean and std
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]


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

        # Do some cleaning on fairface_df
        
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
            self.transform = transforms.Compose(
                [
                    # Resize to 256x256 as specified by DEX
                    transforms.Resize((256, 256)),
                    # Random crop to 224x224
                    transforms.RandomCrop(224),
                    # Randomly mirror the image
                    transforms.RandomHorizontalFlip(),
                    # Convert to the PIL.Image object to tensor (required)
                    transforms.ToTensor(),
                    # Normalize to ImageNet mean
                    transforms.Normalize(mean=imagenet_mean, std=[1.0, 1.0, 1.0]),
                ]
            )
        else:
            self.transform = transforms.Compose(
                [
                    # Resize to 256x256 as specified by DEX
                    transforms.Resize((256, 256)),
                    # Random crop to 224x224
                    transforms.CenterCrop(224),
                    # Convert to the PIL.Image object to tensor (required)
                    transforms.ToTensor(),
                    # Normalize to ImageNet mean
                    transforms.Normalize(mean=imagenet_mean, std=[1.0, 1.0, 1.0]),
                ]
            )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index) -> tuple[Image.Image, int]:
        image_path = self.labels.iloc[index]["full_path"]
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        label = self.labels.iloc[index]["age"]
        return image, label


def main():
    print(FairFaceDataset("data/fairface")[0])


if __name__ == "__main__":
    main()
