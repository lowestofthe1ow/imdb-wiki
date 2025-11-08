import pandas as pd
import torch
import torchvision.transforms as transforms

from torch.utils.data import Dataset
from PIL import Image

# Known values for the ImageNet dataset mean and std
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]


class IMDBWIKIDataset(Dataset):
    """Torch Dataset object for IMDB-WIKI"""

    def __init__(self, csv):
        """Initializes the dataset

        Args:
            csv:    Path to the CSV file containing the IMDB-WIKI image metadata.
        """

        # Load data from CSV file
        self.labels = pd.read_csv(csv)
        self.transform = transforms.Compose(
            [
                # Resize to 256x256 as specified by DEX
                transforms.Resize((256, 256)),
                # Convert to the PIL.Image object to tensor (required)
                transforms.ToTensor(),
                # Normalize to ImageNet mean
                transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
            ]
        )

    def __len__(self):
        """Returns the size of the dataset"""
        return len(self.labels)

    def __getitem__(self, index):
        # We convert to RGB colorspace to match image channels
        image = Image.open(self.labels.iloc[index]["full_path"]).convert("RGB")
        label = self.labels.iloc[index]["age"]

        # Apply the transformations to the PIL image
        try:
            image = self.transform(image)
        except Exception as e:
            print(self.labels.iloc[index]["full_path"])

        return image, label
