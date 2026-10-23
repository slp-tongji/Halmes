from typing import Literal, Sequence

from li_llm_messages import LlmAssistantMessage

from li_openai.openai_llm_parameter import OpenaiLlmParameter
from li_openai.vllm_openai_llm import VllmOpenaiLlm


class VllmInternS1OpenaiLlm(VllmOpenaiLlm):
    def __init__(self, 
                 model: str,
                 api_key: str, 
                 base_url: str,
                 parameter: OpenaiLlmParameter = {},
                 retry_mapping: bool | Literal["respect_temperature"] = "respect_temperature",
                 retry_structured: int = 3,
                 retry: Sequence[float] = (1, 2, 4)):
        parameter = parameter.copy()

        # https://huggingface.co/internlm/Intern-S1-mini#switching-between-thinking-and-non-thinking-modes
        if parameter.get("reasoning_effort", None) == "none":
            extra_body: object = parameter.setdefault("extra_body", {})
            if not isinstance(extra_body, dict):
                raise RuntimeError("Failed to disable thinking as the given parameter includes a non-dictionary body.")
            chat_template_kwargs: object = extra_body.setdefault("chat_template_kwargs", {})
            if not isinstance(chat_template_kwargs, dict):
                raise RuntimeError("Failed to disable thinking as the given parameter includes a non-dictionary chat_template_kwargs.")
            chat_template_kwargs["enable_thinking"] = False

        super().__init__(model, api_key, base_url, parameter, retry_mapping, retry_structured, retry)

    def _convert_response(self, response: str) -> LlmAssistantMessage:
        response_split: list[str] = response.split("\n </think> \n\n", maxsplit=1)
        if len(response_split) == 1:
            response_split: list[str] = response.split("</think>", maxsplit=1)

        if len(response_split) == 1:
            return LlmAssistantMessage(response_split[0], None)
        return LlmAssistantMessage(response_split[1], response_split[0])
    