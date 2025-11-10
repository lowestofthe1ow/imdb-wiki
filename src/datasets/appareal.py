from pathlib import Path
import pandas as pd
import torch
import torchvision.transforms as transforms

from torch.utils.data import Dataset
from PIL import Image

# Known values for the ImageNet dataset mean and std
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]


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

        appareal_train_df["file_name"] = (
            Path(appareal_path) / "train" / appareal_train_df["file_name"]
        )
        appareal_test_df["file_name"] = (
            Path(appareal_path) / "test" / appareal_test_df["file_name"]
        )
        appareal_valid_df["file_name"] = (
            Path(appareal_path) / "valid" / appareal_valid_df["file_name"]
        )

        self.labels = pd.concat(
            [appareal_train_df, appareal_test_df, appareal_valid_df]
        )
        self.labels["apparent_age_avg"] = (
            self.labels["apparent_age_avg"].round().astype("int")
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
        """"""
        return len(self.labels)

    def __getitem__(self, index) -> tuple[Image.Image, int]:
        """
        Returns tensor representation of the Image and apparent age given index
        ! We assume the folder to the images is in the same folder as the csv_path provided
        """
        file_name = self.labels.iloc[index]["file_name"]
        image = Image.open(file_name).convert("RGB")
        label = self.labels.iloc[index]["apparent_age_avg"]

        try:
            image = self.transform(image)
        except Exception as e:
            print(f"Error opening `{file_name}`:\n{e}")

        return image, label


def main():
    print(APPAREALDataset("data/appa-real-release")[0])


if __name__ == "__main__":
    main()
