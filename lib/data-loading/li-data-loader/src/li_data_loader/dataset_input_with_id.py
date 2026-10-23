from typing import NamedTuple, Sequence

from li_llm_messages import LlmMessage


class DatasetInputWithId(NamedTuple):
    id: str
    input: Sequence[LlmMessage]
