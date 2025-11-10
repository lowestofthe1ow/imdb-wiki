import torch
import lightning as L

from src.modules.mean_variance_loss import MeanVarianceLoss
from torch.optim.lr_scheduler import StepLR


class FinetuneMVL(L.LightningModule):
    def __init__(self, model):
        """Initializes the module.

        Args:
            model:  The base model object to fine-tune
        """

        super().__init__()

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
        # MVL, like DEX, uses stochastic gradient descent for optimization
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=0.0001,
            weight_decay=0.0005,
            momentum=0.9,
        )

        # Multiply learning rate by 0.1 every 15 epochs (based on MVL paper)
        scheduler = StepLR(
            optimizer,
            step_size=15,
            gamma=0.1,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }
