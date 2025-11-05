import torch
import lightning as L

from torch.optim.lr_scheduler import StepLR


class FinetuneDEX(L.LightningModule):
    def __init__(self, layer_params, model):
        super().__init__()

        self.layer_params = layer_params
        self.model = model

        # Cross-entropy loss for classification
        self.criterion = torch.nn.CrossEntropyLoss()

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
        # I'm pretty sure stochastic gradient descent is standard here
        optimizer = torch.optim.SGD(
            self.layer_params,
            weight_decay=0.0005,
            momentum=0.9,
        )

        # "The learning rate is reduced every 10 passes through the entire data by a factor of 10"
        scheduler = StepLR(
            optimizer,
            step_size=10,
            gamma=0.1,  # Divide by 10 lol
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }
