import torch
from statictorch import Tensor0d, Tensor1d, Tensor2d

from li_halmes_model.modules.halmes_multi_statistics_aggregator import HalmesMultiStatisticsAggregator
from li_halmes_model.modules.halmes_res_block import HalmesResBlock


class HalmesMlpBackbone(torch.nn.Module):
    def __init__(self, 
                 in_features: int, 
                 hidden_features: int, 
                 layers: int,
                 dropout: float, 
                 tail_length: int):
        super().__init__()
        self.input_norm = torch.nn.LayerNorm(in_features)
        self.aggregator = HalmesMultiStatisticsAggregator(tail_length)

        layers_: list[torch.nn.Module] = [
            torch.nn.Linear(self.aggregator.get_out_features(in_features), hidden_features),
            torch.nn.LayerNorm(hidden_features),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
        ]
        for _ in range(layers - 1):
            layers_.append(HalmesResBlock(hidden_features, dropout))
        layers_.append(torch.nn.Linear(hidden_features, 1))

        self.net = torch.nn.Sequential(*layers_)

    def forward(self, features: Tensor2d) -> Tensor0d:
        x: Tensor2d = self.input_norm(features)
        agg: Tensor1d = self.aggregator(x)
        return Tensor0d(self.net(agg).squeeze())
    