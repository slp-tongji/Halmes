import abc
from typing import Sequence

from li_llm_messages import LlmAssistantMessage, LlmMessage
from transformers.generation.configuration_utils import GenerationConfig


class SupportedLlm(abc.ABC):
    def _copy_config(self, config: GenerationConfig) -> GenerationConfig:
        return GenerationConfig.from_dict(config.to_diff_dict())

    def __init__(self, variant_id: str) -> None:
        super().__init__()
        self.__variant_id = variant_id

    @property
    def variant_id(self) -> str:
        return self.__variant_id

    def chat(self, messages: Sequence[LlmMessage], max_new_tokens: int) -> LlmAssistantMessage:
        ...