import torch
import lightning as L

from torch.optim.lr_scheduler import StepLR


class FinetuneDEX(L.LightningModule):
    """
    PyTorch Lightning module for fine-tuning a model using DEX configurations.
    """

    def __init__(self, layer_params, model):
        """Initializes the module.

        Args:
            layer_params:   Dict of parameter ("layer") groups with specific
                            configurations for each. In the case of DEX, learning rate
                            is 0.0001 for all layers except the last, which has it set
                            to 0.001
            model:          The base model object to fine-tune
        """
        super().__init__()

        self.layer_params = layer_params
        self.model = model

        # Cross-entropy loss for classification
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        """Performs a forward pass through the model"""

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
        # DEX uses stochastic gradient descent for optimization
        optimizer = torch.optim.SGD(
            self.layer_params,
            weight_decay=0.0005,
            momentum=0.9,
        )

        # Multiply learning rate by 0.1 every 10 epochs
        scheduler = StepLR(
            optimizer,
            step_size=10,
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
