# train_dex_imdbwiki.py
# Begins training the model on the IMDB-WIKI dataset based on DEX or MVL

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
from src.datasets.imdbwiki import IMDBWIKIDataset
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
parser.add_argument("--checkpoint-path", type=str)
args = parser.parse_args()


# Load the IMDB-WIKI dataset
dataset = IMDBWIKIDataset("data/imdbwiki.csv")

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

print("Data loaded from /data/imdbwiki.csv.")


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

# Initialize PyTorch Lightning module
match args.method:
    case "DEX":
        model = FinetuneDEX(model=vgg16)
    case "MVL":
        model = FinetuneMVL(model=vgg16)


# Set up saving model checkpoints
os.makedirs("checkpoints", exist_ok=True)
checkpoint_config = ModelCheckpoint(
    dirpath="checkpoints",
    filename=f"IMDBWIKI_{args.method}_{{epoch:02d}}_{{val_loss:.2f}}",
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

if args.checkpoint_path:
    print(".ckpt file provided. Preparing for further training...")
    trainer.fit(
        model=model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=args.checkpoint_path,
    )
else:
    print("No .ckpt file provided. Training from default weights...")
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)


# Save the final model state dict (.pth file) to be safe.
torch.save(model.state_dict(), "checkpoints/imdbwiki_dex.pth")
