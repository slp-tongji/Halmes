from typing import TypedDict

from statictorch import Tensor1d


class HalmesModelState(TypedDict):
    backbone_in_features: int
    backbone_hidden_features: int
    backbone_layers: int
    backbone_dropout: float
    backbone_tail_length: int
    backbone_state: dict

    prior_mu: Tensor1d
    prior_var: Tensor1d