# train_dex_imdbwiki.py
# Begins training the model on the IMDB-WIKI dataset based on DEX or MVL

import argparse
import lightning as L
import numpy as np
import os
import torch

from itertools import chain
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import models

from src.datasets.imdbwiki import IMDBWIKIDataset
from src.datasets.appareal import APPAREALDataset
from src.datasets.chalearn2015 import ChaLearn2015Dataset
from src.modules.finetune_dex import FinetuneDEX
from src.modules.mean_variance_loss import MeanVarianceLoss

# Possible tradeoff between FP precision and performance
torch.set_float32_matmul_precision("medium")

# Command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--batch-size", type=int)
parser.add_argument("--workers", type=int)
parser.add_argument("--min-epochs", type=int)
parser.add_argument("--max-epochs", type=int)
parser.add_argument("--method", type=str)
parser.add_argument("--checkpoint-path", type=str)
parser.add_argument("--dataset", default="IMDBWIKI", type=str)
parser.add_argument("--dataset-path", type=str)
parser.add_argument("--outfile-prefix", default="model", type=str)
parser.add_argument("--use-weighted-ce", action=argparse.BooleanOptionalAction)
args = parser.parse_args()

# Load the dataset
DatasetClass = None
match args.dataset:
    case "IMDBWIKI":
        DatasetClass = IMDBWIKIDataset
    case "APPAREAL":
        DatasetClass = APPAREALDataset
    case "CHALEARN":
        DatasetClass = ChaLearn2015Dataset

dataset = DatasetClass(args.dataset_path)

# Do a 90/10 split for training and validation
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Set the validation subset to use a different configuration
# This configuration uses different transforms for the image inputs
val_dataset.dataset = DatasetClass(args.dataset_path, train=False)

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

print("> Data loaded from /data/imdbwiki.csv.")


# Calculate the weights for each age class if using weighted cross-entropy loss
age_classes = np.arange(0, 101, 1)

if args.use_weighted_ce:
    age_labels = train_dataset.dataset.labels.iloc[train_dataset.indices]["age"].values
    print(age_labels)
    weights = compute_class_weight(
        class_weight="balanced", classes=age_classes, y=age_labels
    )
    print(f"Using weighted CE loss with weights {weights}")
else:
    weights = np.full(101, 1 / 101)  # Uniform weights
    print("Using unweighted CE loss")

weights_tensor = torch.from_numpy(weights).float().to("cuda")


vgg16 = models.vgg16(weights=models.VGG16_Weights.DEFAULT)

# Change output layer to have 101 neurons, one for each age class
vgg16.classifier[6] = torch.nn.Linear(
    4096, len(age_classes)  # 4096 inputs to the final fully connected layer
)

# Initialize PyTorch Lightning module, using CE loss or MVL
match args.method:
    case "DEX":
        model = FinetuneDEX(
            model=vgg16, criterion=torch.nn.CrossEntropyLoss(weight=weights_tensor)
        )
    case "MVL":
        model = FinetuneDEX(
            model=vgg16, criterion=MeanVarianceLoss(0.2, 0.05, weights_tensor)
        )


# Set up saving model checkpoints
os.makedirs("checkpoints", exist_ok=True)
checkpoint_config = ModelCheckpoint(
    dirpath="checkpoints",
    filename=f"{args.outfile_prefix}_{{epoch:02d}}_{{val_loss:.2f}}",
    save_top_k=1,
    monitor="val_loss",
    mode="min",
)


print("## Starting training...")

# Initialize PyTorch Lightning Trainer and begin training
trainer = L.Trainer(
    callbacks=[EarlyStopping(monitor="val_loss", mode="min"), checkpoint_config],
    min_epochs=args.min_epochs,
    max_epochs=args.max_epochs,
)

if args.checkpoint_path:
    print("* .ckpt file provided. Preparing for further training...")
    trainer.fit(
        model=model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=args.checkpoint_path,
    )
    """
    checkpoint = torch.load(args.checkpoint_path, map_location=model.device)[
        "state_dict"
    ]
    model.load_state_dict(checkpoint, strict=False)
    """
else:
    print("* No .ckpt file provided. Training from default weights...")
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)

# Save the final model state dict (.pth file) to be safe.
torch.save(model.state_dict(), "checkpoints/imdbwiki_dex.pth")
