import re
from typing import Iterable, Literal
import typing

from li_bingo_input_provider import BingoInputsSub
from li_data_loader import DatasetJudge
from li_image_operating import InMemoryImage
from li_llm_messages import LlmAssistantMessage, LlmUserMessage
from li_logging import TomlLogLike
from li_openai import OpenaiLlm

from li_bingo_judge.bingo_judgement import BingoJudgement


class BingoJudgeSub(DatasetJudge[BingoJudgement]):
    # https://github.com/gzcch/Bingo/blob/6fa5dfd80236fdb82cd055f02ad5ee6a375f95f3/Bingo_benchmark/bingo_gpt_eval.py
    def __init__(self, 
                 inputs: BingoInputsSub, 
                 judge: OpenaiLlm, 
                 similarity_threshold: Literal[1, 2, 3, 4, 5] = 4) -> None:
        super().__init__()
        self.__inputs = inputs
        self.__judge = judge
        self.__similarity_threshold = similarity_threshold
    
    @property
    def ids(self) -> Iterable[str]:
        return self.__inputs.ids
    
    def __result_mapper(self, message: LlmAssistantMessage) -> Literal[1, 2, 3, 4, 5]:
        match: re.Match | None = re.search(r'\b[1-5](?:\.\d+)?\b', message.content.split('Similarity:')[1])
        if match is None:
            raise RuntimeError("No similarity score found in text")
        return typing.cast(Literal[1, 2, 3, 4, 5], int(match.group(0)))

    def judge(self, id: str, response: LlmAssistantMessage, log: TomlLogLike) -> BingoJudgement:
        with log:
            log.set_document_value(id, "data_id")
            log.set_document_value(response.content, "response")

            image: InMemoryImage
            question: str
            answer: str
            image, question, answer = self.__inputs.get_image_question_and_answer(id)
            log.set_document_value(question, "question")
            log.set_document_value(answer, "correct_answer")

            similarity: int = self.__judge.chat([
                LlmUserMessage(
                    (
                        f"""
        Given a question about an image, there is a correct answer to the question and an answer to be determined. If the correctness of the correct answer is given a score of 5, please rate the correctness of the answer to be determined(1 to 5 points).
        
        Question:
        - question about the image: {question}\n
        
        Answers:
        - correct answer(ground truth): {answer}\n
          answer to be determined: {response.content}\n
          
        Task:\n
        - Given a question about an image, there is a correct answer to the question and an answer to be determined. If the correctness of the correct answer is given a score of 5, please rate the correctness of the answer to be determined(1 to 5 points).
        
        Output Format:
        Similarity: your answer\n
        """, 
                        image
                    )
                )
            ], log.create_sub_log("judge_chat"), self.__result_mapper)

            log.set_document_value(similarity, "similarity")
            
            if similarity < self.__similarity_threshold:
                result: bool = False
                reason: str = f"The similarity '{similarity}' is lower than the threshold '{self.__similarity_threshold}'."
            else:
                result = True
                reason = ""
            
            log.set_document_value(result, "result")
            log.set_document_value(reason, "reason")
            return BingoJudgement(result, reason, similarity)
