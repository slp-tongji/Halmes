from typing import Literal, Sequence

from li_openai.openai_llm import OpenaiLlm
from li_openai.openai_llm_parameter import OpenaiLlmParameter


class VllmOpenaiLlm(OpenaiLlm):
    def __init__(self, 
                 model: str,
                 api_key: str, 
                 base_url: str,
                 parameter: OpenaiLlmParameter = {},
                 retry_mapping: bool | Literal["respect_temperature"] = "respect_temperature",
                 retry_structured: int = 3,
                 retry: Sequence[float] = (1, 2, 4)):
        if retry_mapping == "respect_temperature":
            if "temperature" in parameter:
                retry_mapping = parameter["temperature"] != 0
            else:
                retry_mapping = True
        self.__parameters = parameter

        super().__init__(model, api_key, base_url, retry, retry_mapping, retry_structured)

    def _get_parameter(self) -> OpenaiLlmParameter:
        return self.__parameters
