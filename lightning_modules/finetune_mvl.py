import torch
import lightning as L

from nn_modules.mean_variance_loss import MeanVarianceLoss


class FinetuneMVL(L.LightningModule):
    def __init__(self, learning_rate, batch_size, model):
        super().__init__()

        # Automatically saves params as hyperparameters
        self.save_hyperparameters(ignore=["model"])
        self.model = model

        # Cross-entropy loss for classification
        # Paper used λ_1 = 0.2 and λ_1 = 0.05
        self.criterion = MeanVarianceLoss(0.2, 0.05)

    def forward(self, x):
        # Do a forward pass through the model
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)

        return loss

    def configure_optimizers(self):
        # Paper uses SGD
        optimizer = torch.optim.SGD(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer
