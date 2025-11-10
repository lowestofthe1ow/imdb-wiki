import pandas as pd
import os
import torch
import torchvision.transforms as transforms

from torch.utils.data import Dataset
from PIL import Image

# Known values for the ImageNet dataset mean and std
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]


class IMDBWIKIDataset(Dataset):
    """Torch Dataset object for IMDB-WIKI"""

    def __init__(self, csv, path_prefix, train=True):
        """Initializes the dataset

        Args:
            csv:            Path to the CSV file containing the IMDB-WIKI metadata.
            path_prefix:    Path to prefix to the image paths in the CSV file
        """

        # Load data from CSV file
        self.image_data = pd.read_csv(csv)

        # Add the prefix to the image path
        self.image_data["full_path"] = self.image_data["full_path"].apply(
            lambda x: os.path.join(path_prefix, x)
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
        """Returns the size of the dataset"""
        return len(self.image_data)

    def __getitem__(self, index):
        # We convert to RGB colorspace to match image channels
        image = Image.open(self.image_data.iloc[index]["full_path"]).convert("RGB")
        label = self.image_data.iloc[index]["age"]

        # Apply the transformations to the PIL image
        try:
            image = self.transform(image)
        except Exception as e:
            print(self.image_data.iloc[index]["full_path"])

        return image, label
