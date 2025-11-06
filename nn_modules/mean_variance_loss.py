import torch

class MeanVarianceLoss(torch.nn.Module):
    def __init__(self, lm1, lm2):
        super().__init__()

        # lm1 (λ_1) and lm2 (λ_2) are "two hyper-parameters, balancing the influencing
        # of individual sub-losses in the joint loss"
        self.lm1 = lm1
        self.lm2 = lm2

        assert torch.cuda.is_available()
        self.age_range = torch.arange(101).float().to("cuda")

        print()

    def forward(self, input, target):
        """Does a forward-pass calculation of the mean-variance loss

        Args:
            input: Output from fully-connected layer
            target: True ages for images (labels)
        """

        # Softmax over the output (should be a fully-connected layer)
        # Predicted probability distribution for each image in a batch (dim=1)
        prob = torch.nn.functional.softmax(input, dim=1)

        # m_i = Σ_{j=1}^K j * p_{i,j}
        # Mean of estimated distribution is softmax expected value like in DEX
        # Broadcast to each image in the batch
        mean = torch.sum(prob * self.age_range, dim=1)

        # Mean loss L_m
        # "...penalizes the difference between the mean of an estimated age distribution
        # and the ground-truth age"
        # L2 distance (MSE loss) between mean and target
        # TODO: Paper halves this mean for some reason I don't fully get
        mean_loss = 0.5 * torch.nn.functional.mse_loss(mean, target.float())

        # Variance loss L_v
        # TODO: v_i = Σ_{j=1}^K p_{i,j} * (j - m_i)^2 ...should be right
        # "...penalizes the dispersion of an estimated age distribution"
        # The variance loss for a batch is the mean of the variances for each estimated
        # distribution
        variance_loss = torch.mean(
            # Unsqueeze to broadcast
            torch.sum(prob * (self.age_range - mean.unsqueeze(1)) ** 2, dim=1)
        )

        # Softmax (cross-entropy) loss L_s
        _softmax_loss = torch.nn.CrossEntropyLoss()
        softmax_loss = _softmax_loss(input, target)

        # Combined loss
        # L = L_s + (λ_1 * L_m) + (λ_2 * L_v)
        return softmax_loss + self.lm1 * mean_loss + self.lm2 * variance_loss

        # ...I hope this works
