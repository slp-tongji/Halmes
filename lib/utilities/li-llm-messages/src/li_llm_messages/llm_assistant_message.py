from typing import NamedTuple


class LlmAssistantMessage(NamedTuple):
    content: str
    reasoning: str | None
