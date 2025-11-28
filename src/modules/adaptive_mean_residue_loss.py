from torch import nn
import torch
import numpy as np
import torch.nn.functional as F


class AdaptiveMeanResidueLoss(nn.Module):
    """Compatibility wrapper for __MeanResidueLossAdaptive that forwards a single combined loss value

    Combines a cross-entropy loss on logits with the adaptive mean-residue
    components from `__MeanResidueLossAdaptive` to yield a single scalar (combined) loss.
    """

    def __init__(
        self, lambda_1, lambda_2, start_age=0, end_age=100, K=6, weights_tensor=None
    ):
        """Initializes the module

        Args:
            lambda_1: hyperparameter that weighs the mean loss (higher -> push towards ground truth age)
            lambda_2: hyperparameter that weighs the residue loss (higher -> concentrate probability mass near ground truth)
            start_age
            end_age
        """
        super().__init__()
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.weights = weights_tensor

        # internal implementation of AMR-Loss retrieved from GitHub
        self._amr = _MeanResidueLossAdaptive(lambda_1, lambda_2, start_age, end_age)

    def forward(self, logits, targets):
        """Return a combined loss"""
        ce_loss = F.cross_entropy(logits, targets, weight=self.weights)
        mean_term, residue_term, batch_avg_K = self._amr(logits, targets)

        # mean loss and residue loss already weighted by lambdas 1 and 2
        # so we just add them
        combined = ce_loss + mean_term + residue_term
        return combined


class _MeanResidueLossAdaptive(nn.Module):
    """Official implementation of the Adaptive Mean-Residue Loss function

    THIS CLASS USES CODE RETRIEVED FROM
    https://github.com/jacobzhaoziyuan/AMR-Loss/blob/main/residue_loss_adaptive_K.py
    """

    def __init__(self, lambda_1, lambda_2, start_age, end_age, K=6):
        super().__init__()
        np.random.seed(2019)  # set by the authors
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.start_age = start_age
        self.end_age = end_age

        # `self.K` seems to be unused; paper proposed "a dynamic Top-K pooling"
        # to "obtain a proper range for centralizing the age distribution"
        self.K = K

    def forward(self, input, target):

        N = input.size()[0]
        target = target.type(torch.FloatTensor).cuda()
        m = nn.Softmax(dim=1)
        p = m(input)

        # mean loss
        a = torch.arange(self.start_age, self.end_age + 1, dtype=torch.float32).cuda()
        mean = torch.squeeze((p * a).sum(1, keepdim=True), dim=1)
        mse = (mean - target) ** 2
        mean_loss = mse.mean() / 2.0

        # residue loss ---- K is not constant
        EPS = 1e-3
        width = self.end_age + 1 - self.start_age
        pos = torch.zeros(N, width).cuda()

        for i in range(N):
            try:
                pos[i, target[i].int()] = 1
            except:
                pos[i, int(target.item())] = 1

        prob_gt = (
            (pos * p).sum(1).unsqueeze(dim=1).cuda()
        )  # probs of the gt. dim 1 is squeezed

        prob_gt = prob_gt.expand(N, width)

        # dtype was originally set to `float`. This was changed to torch.float as
        # Pylance reported type issues
        pos_no_K = torch.tensor(p < prob_gt, dtype=torch.float).cuda()

        p_not_K = pos_no_K * p
        residue_loss = (
            (-(p_not_K + EPS) * torch.log(p_not_K + EPS)).sum(1, keepdim=True).mean()
        )

        batch_average_K = float(((pos_no_K == 0).sum() / N).item())

        return self.lambda_1 * mean_loss, self.lambda_2 * residue_loss, batch_average_K
