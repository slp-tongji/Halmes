from typing import Iterable, Mapping

from li_llm_messages import LlmAssistantMessage
from li_logging import TomlLogLike
from li_mmhalbench_judge import MmhalbenchJudge, MmhalbenchJudgement

from li_halmes_score_inference_outputs.supported_judge import SupportedJudge


class WrappedMmhalbenchJudge(SupportedJudge):
    def __init__(self, judge: MmhalbenchJudge) -> None:
        super().__init__()
        self.__judge = judge

    @property
    def ids(self) -> Iterable[str]:
        return self.__judge.ids

    def get_database_id(self, id: str) -> str:
        if id not in self.__judge.ids:
            raise ValueError(f"Cannot deal with '{id}'.")
        return "mmhalbench"

    def judge(self, id: str, response: LlmAssistantMessage, log: TomlLogLike) -> Mapping[str, str | float | int]:
        judgement: MmhalbenchJudgement = self.__judge.judge(id, response, log)

        return {
            "hal": judgement.score <= 2,
            "score": judgement.score
        }
    