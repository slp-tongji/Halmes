from pathlib import Path
from typing import Iterable, Sequence

from li_data_loader import DatasetInputProvider
from li_llm_messages import LlmMessage

from li_amber_input_provider.amber_inputs_discriminative import AmberInputsDiscriminative
from li_amber_input_provider.amber_inputs_generative import AmberInputsGenerative


class AmberInputs(DatasetInputProvider):
    def __init__(self, path_image: Path, path_data: Path) -> None:
        super().__init__()
        
        self.__generative = AmberInputsGenerative(path_image, path_data / "query" / "query_generative.json")
        self.__discriminative_attribute = AmberInputsDiscriminative(
            path_image, 
            path_data / "query" / "query_discriminative-attribute.json")
        self.__discriminative_existence = AmberInputsDiscriminative(
            path_image, 
            path_data / "query" / "query_discriminative-existence.json")
        self.__discriminative_relation = AmberInputsDiscriminative(
            path_image, 
            path_data / "query" / "query_discriminative-relation.json")
        
        self.__merged = self.__generative.merge_many([self.__discriminative_attribute,
                                                      self.__discriminative_existence,
                                                      self.__discriminative_relation])
    
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

    def get(self, id: str) -> Sequence[LlmMessage]:
        return self.__merged.get(id)
    