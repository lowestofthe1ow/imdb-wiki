import numpy as np
import torch
import lightning as L

from torch.optim.lr_scheduler import StepLR


class FinetuneDEX(L.LightningModule):
    """
    PyTorch Lightning module for fine-tuning a model using DEX configurations.
    """

    # Default weights = even distribution across 101 classes
    def __init__(self, model, class_weights=np.full(101, 1 / 101)):
        """Initializes the module.

        Args:
            model:  The base model object to fine-tune
        """
        super().__init__()

        self.model = model

        # Made to replicate the original paper's configurations:
        # https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/age_train.prototxt
        self.layer_params = [
            {  # All feature extraction layers have 1e-4 learning rate
                "params": [p for p in model.features.parameters()],
                "lr": 0.0001,
                "weight_decay": 0.0005,
            },
            {  # All classification layers except the last have 1e-4 learning rate
                "params": [
                    p for i in range(6) for p in model.classifier[i].parameters()
                ],
                "lr": 0.0001,
                "weight_decay": 0.0005,
            },
            {  # Weights for last layer has 1e-3 learning rate
                "params": [model.classifier[6].weight],
                "lr": 0.001,
                "weight_decay": 0.0005,
            },
            {  # Weights for last layer has 2e-3 learning rate and weight decay set to 0
                "params": [model.classifier[6].bias],
                "lr": 0.002,
                "weight_decay": 0.0,
            },
        ]

        weights_tensor = torch.from_numpy(class_weights).float().to("cuda")

        print(f"Using weighted CE loss with weights {class_weights}!")

        # Cross-entropy loss for classification
        self.criterion = torch.nn.CrossEntropyLoss(weight=weights_tensor)

        # Replicate Caffe behavior
        torch.nn.init.normal_(model.classifier[6].weight, mean=0.0, std=0.01)
        torch.nn.init.constant_(model.classifier[6].bias, 0.0)

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
