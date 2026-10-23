import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import csfile
from li_amber_input_provider import AmberInputs
from li_data_loader import DatasetJudge, SimpleDatasetJudgement
from li_llm_messages import LlmAssistantMessage
from li_logging import TomlLogLike
from li_nltk_wrapper import CrossProcessNltkWrapper
from li_openai import OpenaiLlm
import spacy

from li_amber_judge.amber_annotation_discriminative import AmberAnnotationDiscriminative
from li_amber_judge.amber_annotation_generative import AmberAnnotationGenerative
from li_amber_judge.amber_judge_discriminative import AmberJugdeDiscriminative
from li_amber_judge.amber_judge_generative import AmberJudgeGenerative
from li_amber_judge.amber_judgement_generative import AmberJudgementGenerative


class AmberJudge(DatasetJudge[SimpleDatasetJudgement]):
    def __init__(self, 
                 path_data: Path, 
                 yes_or_no_judge: OpenaiLlm,
                 spacy_encoreweblg: spacy.language.Language,
                 nltk: CrossProcessNltkWrapper,
                 chair_threshold: float = 0.2) -> None:
        super().__init__()
        
        annotations: Sequence[AmberAnnotationGenerative | AmberAnnotationDiscriminative]
        annotations = json.loads(csfile.read_all_text(path_data / "annotations.json"))
        relation: Mapping[str, Sequence[str]] = json.loads(csfile.read_all_text(path_data / "relation.json"))
        safe_words: set[str] = set(csfile.read_all_lines(path_data / "safe_words.txt"))
        safe_words.discard("")

        inputs: AmberInputs = AmberInputs(Path(), path_data)

        self.__generative = AmberJudgeGenerative(annotations, inputs.generative, 
                                                 relation, safe_words, nltk, spacy_encoreweblg, chair_threshold)
        self.__discriminative_attribute = AmberJugdeDiscriminative(annotations, inputs.discriminative_attribute, yes_or_no_judge)
        self.__discriminative_existence = AmberJugdeDiscriminative(annotations, inputs.discriminative_existence, yes_or_no_judge)
        self.__discriminative_relation = AmberJugdeDiscriminative(annotations, inputs.discriminative_relation, yes_or_no_judge)

        self.__merged = DatasetJudge.empty().merge_many([
            self.__generative,
            self.__discriminative_attribute,
            self.__discriminative_existence,
            self.__discriminative_relation
        ])
    
    @property
    def generative(self):
        return self.__generative
        
    @property
    def discriminative_attribute(self):
        return self.__discriminative_attribute
        
    @property
    def discriminative_existence(self):
        return self.__discriminative_existence
        
    @property
    def discriminative_relation(self):
        return self.__discriminative_relation
    
    @property
    def ids(self) -> Iterable[str]:
        return self.__merged.ids

    def judge(self, id: str, response: LlmAssistantMessage, log: TomlLogLike) -> SimpleDatasetJudgement:
        return self.__merged.judge(id, response, log)
    