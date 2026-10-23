from statictorch import Tensor0d
import torch


class HalmesFocalBceLoss(torch.nn.Module):
    def __init__(self, 
                 gamma: float = 2.0, 
                 label_smoothing: float = 0.05,
                 weight_positive: float = 1.0):
        super().__init__()
        self.__gamma = gamma
        self.__label_smoothing = label_smoothing
        self.__weight_positive = weight_positive

    def forward(self, logit: Tensor0d, label: bool) -> Tensor0d:
        target: float = 1. if label else 0.
        target = target * (1 - 2 * self.__label_smoothing) + self.__label_smoothing
        weight: float = self.__weight_positive if label else 1.0

        bce: torch.Tensor = torch.nn.functional.binary_cross_entropy_with_logits(
            logit, torch.tensor(target, dtype=logit.dtype, device=logit.device))

        if self.__gamma > 0:
            probability: Tensor0d = Tensor0d(logit.sigmoid())
            probability = Tensor0d(probability if label else 1 - probability)
            bce = bce * (1 - probability) ** self.__gamma

        return Tensor0d(bce * weight)
    