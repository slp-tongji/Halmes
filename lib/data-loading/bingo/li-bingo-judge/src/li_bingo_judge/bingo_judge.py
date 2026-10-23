from typing import Iterable, Literal

from li_bingo_input_provider import BingoInputs
from li_data_loader import DatasetJudge
from li_llm_messages import LlmAssistantMessage
from li_logging import TomlLogLike
from li_openai import OpenaiLlm

from li_bingo_judge.bingo_judge_sub import BingoJudgeSub
from li_bingo_judge.bingo_judgement import BingoJudgement


class BingoJudge(DatasetJudge[BingoJudgement]):
    def __init__(self, 
                 inputs: BingoInputs, 
                 judge: OpenaiLlm, 
                 similarity_threshold: Literal[1, 2, 3, 4, 5] = 4) -> None:
        super().__init__()

        self.__factual = BingoJudgeSub(inputs.factual, judge, similarity_threshold)
        self.__i2i = BingoJudgeSub(inputs.i2i, judge, similarity_threshold)
        self.__ocr = BingoJudgeSub(inputs.ocr, judge, similarity_threshold)
        self.__region = BingoJudgeSub(inputs.region, judge, similarity_threshold)
        self.__t2i = BingoJudgeSub(inputs.t2i, judge, similarity_threshold)

        self.__merged = self.__factual.merge_many([self.__i2i,
                                                   self.__ocr,
                                                   self.__region,
                                                   self.__t2i])
    
    @property
    def factual(self):
        return self.__factual
    
    @property
    def i2i(self):
        return self.__i2i
    
    @property
    def ocr(self):
        return self.__ocr
    
    @property
    def region(self):
        return self.__region
    
    @property
    def t2i(self):
        return self.__t2i

    @property
    def ids(self) -> Iterable[str]:
        return self.__merged.ids
    
    def judge(self, id: str, response: LlmAssistantMessage, log: TomlLogLike) -> BingoJudgement:
        return self.__merged.judge(id, response, log)
    