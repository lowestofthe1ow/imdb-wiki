import torch


class MeanVarianceLoss(torch.nn.Module):
    """Torch NN loss function module that implements mean-variance loss"""

    def __init__(self, lm1, lm2, weights_tensor):
        """Initializes the module

        Args:
            lm1:    The λ_1 hyperparameter, used to weigh the mean loss
            lm2:    The λ_2 hyperparameter, used to weigh the variance loss
        """
        super().__init__()

        self.lm1 = lm1
        self.lm2 = lm2

        self.weights_tensor = weights_tensor

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

        # --------------------

        # TODO: This implementation uses code from:
        # https://www.github-zh.com/projects/199231717-ageestimation

        N = input.size()[0]
        fl_target = target.type(torch.FloatTensor).cuda()
        m = torch.nn.Softmax(dim=1)
        p = m(input)

        # Mean loss
        a = torch.arange(101, dtype=torch.float32).cuda()
        mean = torch.squeeze((p * a).sum(1, keepdim=True), dim=1)
        mse = (mean - fl_target) ** 2
        mean_loss = mse.mean() / 2.0

        # Variance loss
        b = (a[None, :] - mean[:, None]) ** 2
        variance_loss = (p * b).sum(1, keepdim=True).mean()

        # --------------------

        # Softmax (cross-entropy) loss L_s
        # Initialize the loss function (CrossEntropyLoss is an object; we need to do
        # this separately)
        _softmax_loss = torch.nn.CrossEntropyLoss(weight=self.weights_tensor)
        # Calculate the softmax loss
        softmax_loss = _softmax_loss(input, target)

        # Combined (weighted) loss
        # L = L_s + (λ_1 * L_m) + (λ_2 * L_v)
        return softmax_loss + self.lm1 * mean_loss + self.lm2 * variance_loss
