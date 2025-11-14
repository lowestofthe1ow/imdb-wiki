import numpy as np
import torch
import lightning as L

from src.modules.mean_variance_loss import MeanVarianceLoss
from src.modules.finetune_dex import FinetuneDEX

class FinetuneMVL(FinetuneDEX):
    """
    PyTorch Lightning module for fine-tuning a model using a modified DEX
    approach that uses Mean Variance Loss instead of the original Cross Entropy loss.
    """

    def __init__(self, model, class_weights=np.full(101, 1 / 101)):
        """Initializes the module.

        Args:
            model:  The base model object to fine-tune
        """
        super().__init__(model, class_weights)
        
        print(f"* Overriding: Using Mean Variance Loss.")

        # Paper used lm1 = 0.2 and lm2 = 0.05
        lm1 = 0.2
        lm2 = 0.05
        weights_tensor = torch.from_numpy(class_weights).float().to("cuda")
        self.criterion = MeanVarianceLoss(lm1, lm2, weights_tensor)
