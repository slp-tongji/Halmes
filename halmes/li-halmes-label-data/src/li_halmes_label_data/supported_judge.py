import abc
from typing import Iterable, NamedTuple

from li_logging import TomlLogLike
from li_llm_messages import LlmAssistantMessage


class SupportedJudge(abc.ABC):
    class Word(NamedTuple):
        word_identifier: str
        character_index: int
        is_hallucination: bool

    @abc.abstractmethod
    def judge(self, id: str, response: LlmAssistantMessage, log: TomlLogLike) -> Iterable[Word]:
        ...