import numpy as np
import torch
import lightning as L

from torch.optim.lr_scheduler import StepLR
from src.modules.finetune_dex import FinetuneDEX
from src.modules.adaptive_mean_residue_loss import AdaptiveMeanResidueLoss


class FinetuneAMRL(FinetuneDEX):
    """
    PyTorch Lightning module for fine-tuning a model using the Adaptive
    Mean-Residue (AMR) Loss.
    """

    def __init__(self, model, class_weights=np.full(101, 1 / 101)):
        super().__init__(model, class_weights)

        print(f"* Overriding: Using Adaptive Mean-Residue Loss.")

        
        lambda_1 = 0.2 # mean loss weight
        lambda_2 = 0.05 # residue loss weight

        weights_tensor = torch.from_numpy(class_weights).float().to("cuda")

        self.criterion = AdaptiveMeanResidueLoss(
            lambda_1,
            lambda_2,
            start_age=0,
            end_age=100,
            K=6,
            weights_tensor=weights_tensor,
        )
