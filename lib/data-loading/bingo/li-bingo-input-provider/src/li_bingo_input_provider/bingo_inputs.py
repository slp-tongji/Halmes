from pathlib import Path
from typing import Iterable, Sequence

from li_data_loader import DatasetInputProvider
from li_llm_messages import LlmMessage

from li_bingo_input_provider.bingo_inputs_sub import BingoInputsSub



class BingoInputs(DatasetInputProvider):
    def __init__(self, path_huggingface: Path) -> None:
        super().__init__()
        
        images: Path = path_huggingface / "images"
        self.__factual = BingoInputsSub(path_huggingface / "Factual.json", images)
        self.__i2i = BingoInputsSub(path_huggingface / "I2I.json", images)
        self.__ocr = BingoInputsSub(path_huggingface / "OCR.json", images)
        self.__region = BingoInputsSub(path_huggingface / "Region.json", images)
        self.__t2i = BingoInputsSub(path_huggingface / "T2I.json", images)

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

    def get(self, id: str) -> Sequence[LlmMessage]:
        return self.__merged.get(id)
    