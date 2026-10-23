import abc
from typing import Sequence, TypedDict

from li_llm_messages import LlmMessage
from statictorch import Tensor1d, Tensor3d
from transformers.generation.configuration_utils import GenerationConfig


class SupportedLlm(abc.ABC):
    class Output(TypedDict):
        mm_token_type_ids: Tensor1d
        sequence: Tensor1d
        attentions: list[list[Tensor3d]]

    def __init__(self, target_id: str) -> None:
        super().__init__()
        self.__id = target_id

    def _copy_config(self, config: GenerationConfig) -> GenerationConfig:
        return GenerationConfig.from_dict(config.to_diff_dict())

    @property
    def target_id(self) -> str:
        return self.__id

    def chat(self, messages: Sequence[LlmMessage], max_new_tokens: int) -> Output:
        ...
