import torch
import lightning as L

from torchvision import models
from torch.utils.data import Dataset, DataLoader
from torch_datasets.imdbwiki import IMDBWIKIDataset
from lightning_modules.finetune import Finetune

# Possible tradeoff between FP precision and performance
# torch.set_float32_matmul_precision("medium")

# Load IMDB-WIKI data
LOADER_BATCH_SIZE = 32
LOADER_WORKERS = 15
dataset = IMDBWIKIDataset("data/imdbwiki.csv")
loader = DataLoader(
    dataset, batch_size=LOADER_BATCH_SIZE, num_workers=LOADER_WORKERS, shuffle=True
)

print("Data loaded from /data/imdbwiki.csv.")

# Initialize model for fine-tuning

# TODO: Set to weights=VGG16_Weights.DEFAULT (?)
vgg16 = models.vgg16(pretrained=True)

# "When training for classification, the output layer is adapted to 101 output neurons
# corresponding to natural numbers from 0 to 100..."
vgg16.classifier[6] = torch.nn.Linear(
    4096,  # 4096 inputs to the final fully connected layer
    101,  # 101 age classes
)

print("Trained model will be saved to /imdb_wiki.ckpt. Starting training...")

model = Finetune(learning_rate=1e-3, batch_size=32, model=vgg16)
trainer = L.Trainer()
trainer.fit(model=model, train_dataloaders=loader)
