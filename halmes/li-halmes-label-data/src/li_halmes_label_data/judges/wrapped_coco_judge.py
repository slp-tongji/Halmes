from li_coco_judge import CocoJudge, CocoJudgement
from li_llm_messages import LlmAssistantMessage
from li_logging import TomlLogLike
from torch import inner

from li_halmes_label_data.supported_judge import SupportedJudge


class WrappedCocoJudge(SupportedJudge):
    def __init__(self, inner: CocoJudge) -> None:
        self.__inner = inner
    
    def judge(self, id: str, response: LlmAssistantMessage, log: TomlLogLike):
        word: str
        character_index: int
        is_not_hallucination: bool
        for word, character_index, is_not_hallucination in self.__inner.judge(id, response, log).words:
            yield SupportedJudge.Word(word, character_index, not is_not_hallucination)
