from typing import Iterable, Mapping

from li_amber_judge import AmberJudgeGenerative, AmberJudgementGenerative
from li_llm_messages import LlmAssistantMessage
from li_logging import TomlLogLike

from li_halmes_score_inference_outputs.supported_judge import SupportedJudge


class WrappedAmberJudge(SupportedJudge):
    def __init__(self, judge: AmberJudgeGenerative) -> None:
        super().__init__()
        self.__judge = judge

    @property
    def ids(self) -> Iterable[str]:
        return self.__judge.ids

    def get_database_id(self, id: str) -> str:
        if id not in self.__judge.ids:
            raise ValueError(f"Cannot deal with '{id}'.")
        return "amber"

    def judge(self, id: str, response: LlmAssistantMessage, log: TomlLogLike) -> Mapping[str, str | float | int]:
        judgement: AmberJudgementGenerative = self.__judge.judge(id, response, log)

        return {
            "mentioned (chair)": len(judgement.all_considered_objects), 
            "mentioned non-truth (chair)": len(judgement.non_truth_objects), 
            
            "dataset truth (cover)": len(judgement.annotation_truth),
            "covered truth (cover)": len(judgement.found_truth),
            
            "dataset hallucination (cog)": len(judgement.annotation_hallu),
            "covered hallucination (cog)": len(judgement.found_hallu),

            "has hallucination (hal)": len(judgement.non_truth_objects) > 0
        }
    