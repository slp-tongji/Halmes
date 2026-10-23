from collections import deque
from pathlib import Path
import stat
from typing import Iterable, Sequence
import typing

from li_halmes_model import HalmesFeature, HalmesFeatureExtractor
from statictorch import Tensor0d, Tensor1d, Tensor2d, Tensor3d, Tensor4d
import torch

from li_halmes_model.modules.halmes_mlp_backbone import HalmesMlpBackbone
from li_halmes_model.halmes_model_state import HalmesModelState
from li_halmes_model.halmes_gate_parameter import HalmesGateParameter


class HalmesModel:
    def __init__(
        self,
        device: torch.device,
        in_features: int,
        hidden_features: int = 256,
        layers: int = 2,
        dropout: float = 0.15,
        tail_length: int = 5,
    ):
        super().__init__()

        self.__in_features = in_features
        self.__hidden_features = hidden_features
        self.__layers = layers
        self.__dropout = dropout
        self.__tail_length = tail_length

        self.__backbone = HalmesMlpBackbone(
            in_features,
            hidden_features,
            layers,
            dropout,
            tail_length,
        )
        self._init_weights()
        self.__backbone.to(device)

        self.__device = device
        self.__mu: Tensor1d | None = None
        self.__var: Tensor1d | None = None

    def set_train(self, train: bool):
        self.__backbone.train(train)

    @property
    def device(self):
        return self.__device

    def _init_weights(self):
        module: torch.nn.Module
        for module in self.__backbone.modules():
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, torch.nn.Conv1d):
                torch.nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)

    def predict_logit(self, features: Tensor2d) -> Tensor0d:
        return self.__backbone(features)

    def get_parameters(self):
        return self.__backbone.parameters()
    
    def update_prior_values(self, features: list[Tensor2d]):
        all: Tensor2d = Tensor2d(torch.cat(typing.cast(list[torch.Tensor], features), dim=0))
        
        self.__mu = Tensor1d(all.mean(dim=0))
        self.__var = Tensor1d(all.var(dim=0, unbiased=False))

    def export(self) -> HalmesModelState:
        if self.__mu is None:
            raise RuntimeError("Call update_prior_values to fill the prior values before exporting the model.")
        assert self.__var is not None

        return {
            "backbone_dropout": self.__dropout,
            "backbone_hidden_features": self.__hidden_features,
            "backbone_in_features": self.__in_features,
            "backbone_layers": self.__layers,
            "backbone_state": self.__backbone.state_dict(),
            "backbone_tail_length": self.__tail_length,
            "prior_mu": self.__mu,
            "prior_var": self.__var
        }

    @staticmethod
    def load(state: HalmesModelState, device: torch.device, eval_mode: bool = True):
        result: HalmesModel = HalmesModel(device, 
                                          state["backbone_in_features"], 
                                          state["backbone_hidden_features"],
                                          state["backbone_layers"],
                                          state["backbone_dropout"],
                                          state["backbone_tail_length"])
        result.__mu = state["prior_mu"]
        result.__var = state["prior_var"]
        result.__backbone.load_state_dict(state["backbone_state"])

        if eval_mode:
            result.set_train(False)

        return result
    
    class HalmesFeatureSelector:
        def __init__(self, 
                    id: str,
                    svar: bool,
                    attention_text: bool,
                    attention_generated_text: bool,
                    text_visual_ratio: bool,
                    outgoing_entropy: bool,
                    var: Iterable[Iterable[bool]]) -> None:
            """
            var: layer * header (e.g. If the model have 3 layers and each layer has 2 attention headers, you should provide `[[L1H1, L1H2], [L2H1, L2H2], [L3H1, L3H2]]`)
            """
            selector: list[bool] = [v_ for v in var for v_ in v]
            selector.append(svar)
            selector.append(attention_text)
            selector.append(attention_generated_text)
            selector.append(text_visual_ratio)
            selector.append(outgoing_entropy)

            self.__selector = tuple(selector)
            self.__id = id
        
        @property
        def id(self):
            return self.__id
        
        def create_selector_tensor(self, device: torch.device):
            return Tensor1d(torch.tensor(self.__selector, dtype=torch.bool, device=device))
        
        def select[T: Tensor1d | Tensor2d](self, feature: T) -> T:
            selector: Tensor1d = self.create_selector_tensor(feature.device)
            return typing.cast(T, feature[..., selector])
        
    class HalmesGate:
        def __init__(self, 
                     mm_token_type_ids: Tensor1d, 
                     mu: Tensor1d, 
                     var: Tensor1d, 
                     model: 'HalmesModel',
                     features: 'HalmesModel.HalmesFeatureSelector',
                     parameter: HalmesGateParameter) -> None:
            super().__init__()
            
            self.__halmes = model
            self.__feature_selector = features
            self.__mm_token_type_ids = mm_token_type_ids
            self.__parameter = parameter

            self.__last = False
            self.__features: deque[Tensor1d] = deque(maxlen=self.__parameter.window * 2)
            self.__scores: list[float] = []

            self.__norm_mu: Tensor1d | None = None
            self.__norm_m2: Tensor1d | None = None
            self.__norm_count: int = 0
            
            self.__warm_up: int = 0
            self.__cool_down: int = 0

            self.__mu = Tensor1d(mu.to(mm_token_type_ids.device))
            self.__var = Tensor1d(var.to(mm_token_type_ids.device))
        
        def next(self, attentions: Sequence[Tensor3d]) -> bool:
            """
            Returns `True` if require intervention.
            """
            features: Tensor1d = HalmesFeatureExtractor.to_single_tensor(
                HalmesFeatureExtractor.extract_single(
                    attentions, 
                    self.__mm_token_type_ids
                )
            )
            features = self.__feature_selector.select(features)
            self.__features.append(features)
            
            if self.__norm_mu is None:
                self.__norm_mu = self.__mu
                self.__norm_m2 = Tensor1d(self.__var * max(self.__parameter.n - 1, 1))
                self.__norm_count = self.__parameter.n
                self.__norm_count += 1
                delta: Tensor1d = Tensor1d(features - self.__norm_mu)
                self.__norm_mu = Tensor1d(self.__norm_mu + delta / self.__norm_count)
                delta2: Tensor1d = Tensor1d(features - self.__norm_mu)
                self.__norm_m2 = Tensor1d(self.__norm_m2 + delta * delta2)
            assert self.__norm_m2 is not None
            
            window_features: Tensor2d = Tensor2d(
                torch.stack(tuple(self.__features)[-self.__parameter.window:], dim=0)
            )
            window_features = Tensor2d(
                (window_features - self.__norm_mu.unsqueeze(0)) / 
                (self.__norm_m2 / max(self.__norm_count - 1, 1)).clamp_min(1e-6).sqrt().unsqueeze(0)
            )
    
            score: float = float(self.__halmes.predict_logit(window_features).sigmoid())
            self.__scores.append(score)
        
            valid_scores: list[float] = [s for s in self.__scores if s > 0]
            if len(valid_scores) < 6:
                trend: float = 0.0
            else:
                recent: list[float] = valid_scores[-self.__parameter.trend_window:]
                half: int = len(recent) // 2
                first_half_mean: float = sum(recent[:half]) / half
                second_half_mean: float = sum(recent[half:]) / (len(recent) - half)
                trend = second_half_mean - first_half_mean

            if self.__last:
                if score < self.__parameter.theta_low:
                    self.__cool_down += 1
                    if self.__cool_down >= self.__parameter.cool_down:
                        self.__last = False
                else:
                    self.__cool_down = 0
            else:
                if score > self.__parameter.theta_high and trend > self.__parameter.trend_threshold:
                    self.__warm_up += 1
                    if self.__warm_up >= self.__parameter.warm_up:
                        self.__last = True
                else:
                    self.__warm_up = 0

            return self.__last
    
    def new_gate(self, features: HalmesFeatureSelector, mm_token_type_ids: Tensor1d, parameter: HalmesGateParameter):
        if self.__mu is None:
            raise RuntimeError("Call update_prior_values to fill the prior values before using the model.")
        assert self.__var is not None
        return HalmesModel.HalmesGate(mm_token_type_ids, self.__mu, self.__var, self, features, parameter)
    