from typing import Sequence

from li_openai.openai_llm import OpenaiLlm
from li_openai.openai_llm_parameter import OpenaiLlmParameter


class RemoteOpenaiLlm(OpenaiLlm):
    def __init__(self, 
                 model: str,
                 api_key: str,
                 base_url: str,
                 parameter: OpenaiLlmParameter = {},
                 retry_mapping: bool = True,
                 retry_structured: int = 3,
                 retry: Sequence[float] = (1, 2, 4, 8, 16)):
        self.__parameter = parameter
        super().__init__(model, api_key, base_url, retry, retry_mapping, retry_structured)

    def _get_parameter(self) -> OpenaiLlmParameter:
        return self.__parameter
