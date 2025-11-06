import torch


class MeanVarianceLoss(torch.nn.Module):
    """Torch NN loss function module that implements mean-variance loss"""

    def __init__(self, lm1, lm2):
        """Initializes the module

        Args:
            lm1:    The λ_1 hyperparameter, used to weigh the mean loss
            lm2:    The λ_2 hyperparameter, used to weigh the variance loss
        """
        super().__init__()

        self.lm1 = lm1
        self.lm2 = lm2

        # Generate age range tensor and move it to CUDA device (i.e. the GPU; required
        # for training)
        assert torch.cuda.is_available()
        self.age_range = torch.arange(101).float().to("cuda")

    def forward(self, input, target):
        """Does a forward-pass calculation of the mean-variance loss

        Args:
            input:  The outputs of the last fully-connected layer (contains a result
                    for each image in the batch), which we will be validating.
            target: True ages for the images in the batch (labels)
        """

        # Softmax over the output (should be a fully-connected layer)
        # Predicted probability distribution for each image in a batch (dim=1)
        prob = torch.nn.functional.softmax(input, dim=1)

        # m_i = Σ_{j=1}^K j * p_{i,j}
        # Mean of estimated distribution is softmax expected value like in DEX
        # Note that this broadcasts to each image in a batch
        mean = torch.sum(prob * self.age_range, dim=1)

        # Mean loss L_m
        # Penalizes the difference between the mean of an estimated age distribution
        # and the ground-truth age, calculated as L2 distance (MSE loss) between mean
        # m_i and target.
        mean_loss = 0.5 * torch.nn.functional.mse_loss(mean, target.float())

        # Variance loss L_v
        # v_i = Σ_{j=1}^K p_{i,j} * (j - m_i)^2
        # Penalizes the dispersion of an estimated age distribution.
        # The variance loss L_v for a batch is the mean of the variances v_i for each
        # image's estimated distribution.
        variance_loss = torch.mean(
            # Unsqueeze to broadcast
            torch.sum(prob * (self.age_range - mean.unsqueeze(1)) ** 2, dim=1)
        )

        # Softmax (cross-entropy) loss L_s
        # Initialize the loss function (CrossEntropyLoss is an object; we need to do
        # this separately)
        _softmax_loss = torch.nn.CrossEntropyLoss()
        # Calculate the softmax loss
        softmax_loss = _softmax_loss(input, target)

        # Combined (weighted) loss
        # L = L_s + (λ_1 * L_m) + (λ_2 * L_v)
        return softmax_loss + self.lm1 * mean_loss + self.lm2 * variance_loss
