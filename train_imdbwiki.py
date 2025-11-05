# train_dex_imdbwiki.py
# Begins training the model on the IMDB-WIKI dataset based on DEX

# Parameters:
# --batch-size  Batch size
# --workers     Number of CPU workers
# --min-epochs  Minimum number of epochs, even if early stop is triggered
# --max-epochs  Maximum number of epochs, even if early stop is NOT triggered
# --method      "MVL" or "DEX"

import torch
import lightning as L
import argparse
import os

from itertools import chain
from torchvision import models
from torch.utils.data import Dataset, DataLoader, random_split
from torch_datasets.imdbwiki import IMDBWIKIDataset
from lightning_modules.finetune_dex import FinetuneDEX
from lightning_modules.finetune_mvl import FinetuneMVL
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint

# Possible tradeoff between FP precision and performance
# torch.set_float32_matmul_precision("medium")

# Command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--batch-size", type=int)
parser.add_argument("--workers", type=int)
parser.add_argument("--min-epochs", type=int)
parser.add_argument("--max-epochs", type=int)
parser.add_argument("--learning-rate", type=float)
parser.add_argument("--method", type=str)
args = parser.parse_args()

# Load IMDB-WIKI data
dataset = IMDBWIKIDataset("data/imdbwiki.csv")

# 90/10 split for training and validation
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(
    train_dataset, batch_size=args.batch_size, num_workers=args.workers, shuffle=True
)

val_loader = DataLoader(
    val_dataset, batch_size=args.batch_size, num_workers=args.workers
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

# "Any fine-tuning the learning rate for all layers except the last layer is set to
# 0.0001 [...] we set the learning rate for the output layer to 0.001"
layer_params = [
    {
        "params": [p for i in range(6) for p in vgg16.classifier[i].parameters()],
        "lr": 0.0001,
    },
    {"params": list(vgg16.classifier[6].parameters()), "lr": 0.001},
]

print("Starting training...")

match args.method:
    case "DEX":
        model = FinetuneDEX(
            model=vgg16,
            layer_params=layer_params,
        )
    case "MVL":
        model = FinetuneMVL(
            model=vgg16,
            layer_params=layer_params,
        )


# Set up saving model checkpoints
os.makedirs("checkpoints", exist_ok=True)
checkpoint_config = ModelCheckpoint(
    dirpath="checkpoints",
    filename="my_model_{epoch:02d}_{val_loss:.2f}",
    save_top_k=1,
    monitor="val_loss",
    mode="min",
)


trainer = L.Trainer(
    callbacks=[EarlyStopping(monitor="val_loss", mode="min"), checkpoint_config],
    min_epochs=args.min_epochs,
    max_epochs=args.max_epochs,
)

trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)

# Save the final model
torch.save(model.state_dict(), "models/imdbwiki_dex.pth")
