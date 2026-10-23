import torch
from statictorch import Tensor0d, Tensor1d, Tensor2d


class HalmesMultiStatisticsAggregator(torch.nn.Module):
    def __init__(self, tail_length: int = 5):
        super().__init__()
        self.__tail_length = tail_length

    def get_out_features(self, in_features: int):
        # mean, tail_mean, std, slope, max -> [5 * D]
        return in_features * 5

    def forward(self, features: Tensor2d):
        length: int
        feature_count: int
        length, feature_count = features.shape

        device: torch.device
        dtype: torch.dtype
        device, dtype = features.device, features.dtype

        features_mean: Tensor0d = Tensor0d(features.mean(dim=0))

        std: Tensor1d = Tensor1d(
            features.std(dim=0) if length > 1 
            else torch.zeros(feature_count, device=device, dtype=dtype)
        )

        if length >= 3:
            x: Tensor1d = Tensor1d(torch.arange(length, device=device, dtype=dtype))
            x_mean: Tensor0d = Tensor0d(x.mean())
            x_var: Tensor0d = Tensor0d(((x - x_mean) ** 2).sum())
            slope: Tensor1d = Tensor1d(
                ((features - features_mean) * (x - x_mean).unsqueeze(1)).sum(dim=0) / (x_var + 1e-8)
            )
        else:
            slope = Tensor1d(torch.zeros(feature_count, device=device, dtype=dtype))

        return Tensor1d(torch.cat([
            features_mean, 
            features[-min(self.__tail_length, length):].mean(dim=0), 
            std, 
            slope, 
            features.max(dim=0).values
        ], dim=0))
    