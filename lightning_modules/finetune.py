import torch
import lightning as L

from lightning.pytorch.callbacks.early_stopping import EarlyStopping


class Finetune(L.LightningModule):
    def __init__(self, learning_rate, batch_size, model):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])  # Saves params as hyperparameters

        # Save model
        self.model = model

        # Cross-entropy loss for classification
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)  # Do a forward pass through the model

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    """
    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)

        return loss
    """

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer
