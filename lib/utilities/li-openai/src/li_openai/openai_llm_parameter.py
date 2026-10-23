from typing import Literal, TypedDict

import openai


class OpenaiLlmParameter(TypedDict, total=False):
    max_completion_tokens: int
    prompt_cache_key: str
    reasoning_effort: Literal["none", "minimal", "low", "medium", "high", "xhigh"]
    seed: int
    temperature: float
    verbosity: Literal["low", "medium", "high"]
    extra_headers: openai._types.Headers
    extra_query: openai._types.Query
    extra_body: openai._types.Body
    