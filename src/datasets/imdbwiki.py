import pandas as pd
import os
import torch
import torchvision.transforms as transforms

from torch.utils.data import Dataset
from PIL import Image

from src.datasets.constants import IMAGENET_MEAN, IMAGENET_STD, TRANSFORM_TO_SIZE, CROP_SIZE

class IMDBWIKIDataset(Dataset):
    """Torch Dataset object for IMDB-WIKI"""

    def __init__(self, csv_path: str, path_prefix: str, train=True):
        """Initializes the dataset

        Args:
            csv_path:       Path to the CSV file containing the IMDB-WIKI metadata.
            path_prefix:    Path to prefix to the image paths in the CSV file
        """
        self.image_data = pd.read_csv(csv_path)
        # self.image_data["full_path"] = self.image_data["full_path"].apply(
        #     lambda x: os.path.join(path_prefix, x)
        # )

        if train:
            self.transform = transforms.Compose(
                [
                    transforms.Resize(TRANSFORM_TO_SIZE),
                    transforms.RandomCrop(CROP_SIZE),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                ]
            )
        else:
            self.transform = transforms.Compose(
                [
                    transforms.Resize(TRANSFORM_TO_SIZE),
                    transforms.CenterCrop(CROP_SIZE),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                ]
            )

    def __len__(self):
        """Returns the size of the dataset"""
        return len(self.image_data)

    def __getitem__(self, index):
        image_path = self.image_data.iloc[index]["full_path"]
        image = Image.open(image_path).convert("RGB")
        label = self.image_data.iloc[index]["age"]

        try:
            image = self.transform(image)
        except Exception as e:
            print(self.image_data.iloc[index]["full_path"])

        return image, label
