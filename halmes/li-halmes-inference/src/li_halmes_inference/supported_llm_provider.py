from typing import Any, Callable, Protocol

from li_halmes_model import HalmesModel
from li_halmes_configurations import HalmesTargetLlm
import torch

from li_halmes_inference.supported_llm import SupportedLlm


class SupportedLlmProvider(Protocol):
    def __call__(self, 
                 target: HalmesTargetLlm, 
                 features: HalmesModel.HalmesFeatureSelector,
                 halmes: Callable[[torch.device], HalmesModel]) -> SupportedLlm:
        ...
    