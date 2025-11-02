import torch
import lightning as L

from torch.utils.data import Dataset, DataLoader
from torch_datasets.imdbwiki import IMDBWIKIDataset
from lightning_modules.imdbwiki_finetune import IMDBWIKIFinetune

# torch.set_float32_matmul_precision('medium')

LOADER_BATCH_SIZE = 32
LOADER_WORKERS = 15
dataset = IMDBWIKIDataset("data/imdbwiki.csv")
loader = DataLoader(
    dataset, batch_size=LOADER_BATCH_SIZE, num_workers=LOADER_WORKERS, shuffle=True
)

print("Data loaded from /data/imdbwiki.csv")

print("Starting training...")

model = IMDBWIKIFinetune(1e-3, 32)
trainer = L.Trainer()
trainer.fit(model=model, train_dataloaders=loader)
