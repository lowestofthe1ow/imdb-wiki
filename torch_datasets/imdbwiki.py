import pandas as pd
import torch
import torchvision.transforms as transforms

from torch.utils.data import Dataset
from PIL import Image


class IMDBWIKIDataset(Dataset):
    def __init__(self, csv, transform=None):
        self.labels = pd.read_csv(csv)

        # No need to transform IMDB-WIKI samples
        self.transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        image = Image.open(self.labels.iloc[0]["full_path"])
        label = self.labels.iloc[0]["age"]

        return self.transform(image), label
