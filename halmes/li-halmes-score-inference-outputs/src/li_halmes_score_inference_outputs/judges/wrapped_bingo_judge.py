from typing import Iterable, Mapping

from li_amber_judge import AmberJudgeGenerative, AmberJudgementGenerative
from li_bingo_judge import BingoJudge, BingoJudgement
from li_llm_messages import LlmAssistantMessage
from li_logging import TomlLogLike

from li_halmes_score_inference_outputs.supported_judge import SupportedJudge


class WrappedBingoJudge(SupportedJudge):
    def __init__(self, judge: BingoJudge) -> None:
        super().__init__()
        self.__judge = judge

    @property
    def ids(self) -> Iterable[str]:
        return self.__judge.ids

    def get_database_id(self, id: str) -> str:
        if id not in self.__judge.ids:
            raise ValueError(f"Cannot deal with '{id}'.")
        return "bingo"

    def judge(self, id: str, response: LlmAssistantMessage, log: TomlLogLike) -> Mapping[str, str | float | int]:
        judgement: BingoJudgement = self.__judge.judge(id, response, log)

        return {
            "score": judgement.similarity
        }
    