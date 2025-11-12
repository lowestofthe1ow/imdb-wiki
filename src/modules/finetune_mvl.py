import torch
import lightning as L

from src.modules.mean_variance_loss import MeanVarianceLoss
from torch.optim.lr_scheduler import StepLR


class FinetuneMVL(L.LightningModule):
    def __init__(self, model, class_weights):
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

        # Convert weights array into a tensor and move to CUDA device
        weights_tensor = torch.from_numpy(class_weights).float().to("cuda")

        # Cross-entropy loss for classification
        # Paper used λ_1 = 0.2 and λ_1 = 0.05
        self.criterion = MeanVarianceLoss(0.2, 0.05, weights_tensor)

        # Replicate Caffe behavior
        torch.nn.init.normal_(model.classifier[6].weight, mean=0.0, std=0.01)
        torch.nn.init.constant_(model.classifier[6].bias, 0.0)

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
