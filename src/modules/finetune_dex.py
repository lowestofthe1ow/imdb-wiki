import torch
import lightning as L

from torch.optim.lr_scheduler import StepLR


class FinetuneDEX(L.LightningModule):
    """
    PyTorch Lightning module for fine-tuning a model using DEX configurations.
    """

    def __init__(self, model, class_weights):
        """Initializes the module.

        Args:
            layer_params:   Dict of parameter ("layer") groups with specific
                            configurations for each. In the case of DEX, learning rate
                            is 0.0001 for all layers except the last, which has it set
                            to 0.001
            model:          The base model object to fine-tune
        """
        super().__init__()

        self.model = model

        self.layer_params = [
            {
                "params": [
                    p for i in range(6) for p in model.classifier[i].parameters()
                ],
                "lr": 0.0001,
                "weight_decay": 0.0005,
            },
            {
                "params": [model.classifier[6].weight],
                "lr": 0.001,
                "weight_decay": 0.0005,
            },
            {
                "params": [model.classifier[6].bias],
                "lr": 0.002,
                "weight_decay": 0.0,
            },
        ]

        # Convert weights array into a tensor and move to CUDA device
        weights_tensor = torch.from_numpy(class_weights).float().to("cuda")

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
        # ...but let's try AdamW
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
