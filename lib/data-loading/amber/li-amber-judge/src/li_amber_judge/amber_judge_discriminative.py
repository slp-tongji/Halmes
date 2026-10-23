from typing import Iterable, Sequence
import pydantic

from li_amber_input_provider import AmberInputsDiscriminative
from li_data_loader import DatasetJudge, SimpleDatasetJudgement
from li_llm_messages import LlmAssistantMessage, LlmSystemMessage, LlmUserMessage
from li_logging import TomlLogLike
from li_openai import OpenaiLlm

from li_amber_judge.amber_annotation_discriminative import AmberAnnotationDiscriminative
from li_amber_judge.amber_annotation_generative import AmberAnnotationGenerative


class AmberJugdeDiscriminative(DatasetJudge[SimpleDatasetJudgement]):
    # AMBER does not provide a method for parsing model responses; here we follow the third approach at
    # https://github.com/junyangwang0410/AMBER/issues/5#issuecomment-2441766981
    def __init__(self, 
                 annotations: Sequence[AmberAnnotationGenerative | AmberAnnotationDiscriminative], 
                 filter: AmberInputsDiscriminative,
                 yes_or_no_judge: OpenaiLlm) -> None:
        super().__init__()
        self.__annotations: dict[str, bool] = {}

        ids: set[str] = set(filter.ids)
        annotation: AmberAnnotationGenerative | AmberAnnotationDiscriminative
        for annotation in annotations:
            id: str = f"amber_{annotation["id"]}"
            if id in ids:
                assert annotation["type"] != "generative"
                assert id not in self.__annotations
                self.__annotations[id] = annotation["truth"] == "yes"

        self.__yes_or_no_judge = yes_or_no_judge
    
    @property
    def ids(self) -> Iterable[str]:
        return self.__annotations.keys()

    class _ActualAnswer(pydantic.BaseModel):
        is_yes: bool | None = pydantic.Field()

    def judge(self, id: str, response: LlmAssistantMessage, log: TomlLogLike) -> SimpleDatasetJudgement:
        with log:
            log.set_document_value(id, "data_id")
            log.set_document_value(response.content, "raw_response")

            response_bool: bool | None = self.__yes_or_no_judge.chat_structured([
                LlmSystemMessage(
                    """
                    Role:
                    You are a response classification assistant. Your task is to analyze an LLM's response and determine if it conveys a definitive "Yes" or "No".

                    Task:
                    Review the provided text and output a JSON object indicating the stance of the response.

                    Classification Rules:
                    { "is_yes": true }: If the response explicitly starts with "Yes," contains a clear affirmative confirmation, or concludes that the answer is "Yes."
                    { "is_yes": false }: If the response explicitly starts with "No," contains a clear negative denial, or concludes that the answer is "No."
                    { "is_yes": null }: If the response is neutral, descriptive without taking a side, or does not contain a clear "Yes" or "No" conclusion.

                    Examples:
                    Response: Yes, the sky in the image is blue. -> { "is_yes": true }
                    Response: The sky in the image is blue rather than red. Therefore, my answer is no. -> { "is_yes": false }
                    Response: The sky in the image is blue. -> { "is_yes": null }
                    """).dedented(),
                LlmUserMessage((f"Response:\n{response.content}",))
            ], AmberJugdeDiscriminative._ActualAnswer, log.create_sub_log("yes_or_no_judge")).is_yes

            log.set_document_value(response_bool, "response_bool")

            truth: bool = self.__annotations[id]
            log.set_document_value(truth, "truth")

            result: bool
            reason: str
            if response_bool == truth:
                result = True
                reason = ""
            elif response_bool is None:
                result = False
                reason = ("This is a discriminative task, "
                          "but the model does not make a clear choice between 'yes' and 'no'.")
            else:
                result = False
                reason = (f"This is a discriminative task. "
                          f"The model's answer is considered as '{response_bool}', "
                          f"which does not match the correct answer '{truth}'.")
            log.set_document_value(result, "result")
            log.set_document_value(reason, "reason")
            return SimpleDatasetJudgement(result, reason)
