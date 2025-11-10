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

    def __init__(self, csv_path: str, train=True):
        """
        Args:
            csv_path:   Path to the CSV file containing the APPA-REAL image metadata
                        Three CSV files -- gt_avg_test.csv, gt_avg_train.csv
                        and gt_avg_valid.csv -- are used for this purpose.
        """
        self.csv_path = csv_path
        self.set_type = csv_path.split("_")[-1].split('.')[0] # test, train, valid
        self.labels = pd.read_csv(csv_path)
        self.labels["apparent_age_avg"] = self.labels["apparent_age_avg"].round().astype("int")
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
        file_name = Path(self.csv_path).parent / self.set_type / self.labels.iloc[index]['file_name']
        image = Image.open(file_name).convert("RGB")
        label = self.labels.iloc[index]["apparent_age_avg"]

        try:
            image = self.transform(image)
        except Exception as e:
            print(f"Error opening `{file_name}`:\n{e}")
        
        return image, label