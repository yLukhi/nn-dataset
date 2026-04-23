import torch


class Net:
    """
    MAE-based metric for age estimation.

    Returns a normalized accuracy-like score:
        score = max(0, 1 - MAE / threshold)

    With threshold=15:
        score = 0.767  -> MAE ~= 3.5 years
        score = 0.733  -> MAE ~= 4.0 years

    Also exposes raw MAE in years through get_mae().
    """

    def __init__(self, max_mae_threshold: float = 15.0):
        self.name = "mae"
        self.max_mae_threshold = float(max_mae_threshold)
        self.reset()

    def reset(self):
        self._total_abs_error = 0.0
        self._total_samples = 0

    def update(self, outputs: torch.Tensor, labels: torch.Tensor):
        outputs = outputs.detach().float().reshape(-1)
        labels = labels.detach().float().reshape(-1)

        if outputs.numel() != labels.numel():
            raise ValueError(
                f"Metric shape mismatch: outputs has {outputs.numel()} values "
                f"but labels has {labels.numel()} values."
            )

        abs_errors = torch.abs(outputs - labels)
        self._total_abs_error += abs_errors.sum().item()
        self._total_samples += outputs.numel()

    def __call__(self, outputs: torch.Tensor, labels: torch.Tensor):
        self.update(outputs, labels)
        return self._total_abs_error, self._total_samples

    def compute(self) -> float:
        """
        Returns normalized accuracy-like score in [0, 1].
        Higher is better.
        """
        if self._total_samples == 0:
            return 0.0

        mae_years = self._total_abs_error / self._total_samples
        score = 1.0 - (mae_years / self.max_mae_threshold)
        return max(0.0, float(score))

    def result(self) -> float:
        return self.compute()

    def get_mae(self) -> float:
        """
        Returns raw MAE in years.
        """
        if self._total_samples == 0:
            return float('inf')
        return float(self._total_abs_error / self._total_samples)


def create_metric(out_shape=None):
    """
    Framework factory function.
    """
    return Net(max_mae_threshold=15.0)