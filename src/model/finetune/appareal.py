# finetune.py
# Finetune a pre-trained model on APPA-REAL.

# Parameters:
# --batch-size         Batch size
# --workers            Number of CPU workers
# --min-epochs         Minimum number of epochs, even if early stop is triggered
# --max-epochs         Maximum number of epochs, even if early stop is NOT triggered
# --method             "MVL" or "DEX"
# --csv-path           Data to train on
# --checkpoint-path    Checkpoint of the model to fine-tune on

from pathlib import Path
import torch
import lightning as L
import argparse
import os

from itertools import chain
from torchvision import models
from torch.utils.data import Dataset, DataLoader, random_split
from src.datasets.appareal import APPAREALDataset
from src.modules.finetune_dex import FinetuneDEX
from src.modules.finetune_mvl import FinetuneMVL
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
parser.add_argument("--csv-path", type=str)
parser.add_argument("--checkpoint-path", type=str)
args = parser.parse_args()


# Load the IMDB-WIKI dataset
dataset = APPAREALDataset(args.csv_path)

# Do a 90/10 split for training and validation
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Create two separate data loaders: one for training and another for validation
train_loader = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    num_workers=args.workers,
    shuffle=True,  # Training data is shuffled, but not validation data
)
val_loader = DataLoader(
    val_dataset,
    batch_size=args.batch_size,
    num_workers=args.workers,
)

print(f"Data loaded from {args.csv_path}.")


# Initialize model for fine-tuning
vgg16 = models.vgg16(weights=models.VGG16_Weights.DEFAULT)

# Change output layer to have 101 neurons, one for each age class
vgg16.classifier[6] = torch.nn.Linear(
    4096,  # 4096 inputs to the final fully connected layer
    101,  # 101 age classes
)

# Freeze feature extraction layers
for param in vgg16.features.parameters():
    param.requires_grad = False

# USED IN DEX ONLY: All layers except the last have a learning rate of 0.0001.
# Last layer has a learning rate of 0.001
# Learning rate is fixed in the MVL paper
layer_params = [
    {
        # Flatten an iterator that goes over all the classifier layers in VGG16
        "params": [p for i in range(6) for p in vgg16.classifier[i].parameters()],
        "lr": 0.0001,
    },
    {"params": list(vgg16.classifier[6].parameters()), "lr": 0.001},
]

# Initialize PyTorch Lightning modules depending on method used
match args.method:
    case "DEX":
        model = FinetuneDEX.load_from_checkpoint(
            args.checkpoint_path,
            model=vgg16,
            layer_params=layer_params,
        )
    case "MVL":
        model = FinetuneMVL.load_from_checkpoint(args.checkpoint_path, model=vgg16)

# Set up saving model checkpoints
os.makedirs("checkpoints", exist_ok=True)
old_checkpoint = Path(args.checkpoint_path).name
checkpoint_config = ModelCheckpoint(
    dirpath="checkpoints",
    filename=f"{old_checkpoint}-FINETUNED-ON_APPAREAL_{args.method}_{{epoch:02d}}_{{val_loss:.2f}}",
    save_top_k=1,
    monitor="val_loss",
    mode="min",
)

print("Starting training...")

# Initialize PyTorch Lightning Trainer and begin training
trainer = L.Trainer(
    callbacks=[EarlyStopping(monitor="val_loss", mode="min"), checkpoint_config],
    min_epochs=args.min_epochs,
    max_epochs=args.max_epochs,
)
trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)
