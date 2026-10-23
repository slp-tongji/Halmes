from typing import AbstractSet, Iterable, Mapping

from li_coco_judge import CocoJudge, CocoJudgement
from li_llm_messages import LlmAssistantMessage
from li_logging import TomlLogLike
import more_itertools

from li_halmes_score_inference_outputs.supported_judge import SupportedJudge


class WrappedCocoJudge(SupportedJudge):
    def __init__(self, judge: CocoJudge) -> None:
        super().__init__()
        self.__judge = judge

    @property
    def ids(self) -> Iterable[str]:
        return self.__judge.ids

    def get_database_id(self, id: str) -> str:
        if id not in self.__judge.ids:
            raise ValueError(f"Cannot deal with '{id}'.")
        return "coco"

    def judge(self, id: str, response: LlmAssistantMessage, log: TomlLogLike) -> Mapping[str, str | float | int]:
        judgement: CocoJudgement = self.__judge.judge(id, response, log)
        mentioned_hallucination: int = more_itertools.ilen(
            word
            for word, _, is_not_hallucination in judgement.words
            if not is_not_hallucination
        )

        return {
            "mentioned (chair i)": len(judgement.words), 
            "mentioned hallucination (chair i)": mentioned_hallucination, 
            
            "has hallucination (chair s)": mentioned_hallucination > 0, 

            "dataset truth (recall)": len(judgement.dataset_true_objects),
            "covered truth (recall)": len(set(
                word
                for word, _, is_not_hallucination in judgement.words
                if is_not_hallucination
            ))
        }
    