import torch
import lightning as L

from torchvision import models
from lightning.pytorch.callbacks.early_stopping import EarlyStopping


class IMDBWIKIFinetune(L.LightningModule):
    def __init__(self, learning_rate, batch_size):
        super().__init__()
        self.save_hyperparameters()  # Saves params as hyperparameters

        # TODO: Set to weights=VGG16_Weights.DEFAULT (?)
        vgg16 = models.vgg16(pretrained=True)

        # "When training for classification, the output layer is adapted to 101 output
        # neurons corresponding to natural numbers from 0 to 100..."
        vgg16.classifier[6] = torch.nn.Linear(
            4096,  # 4096 inputs to the final fully connected layer
            101,  # 101 age classes
        )

        # Save model
        self.model = vgg16

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

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer
