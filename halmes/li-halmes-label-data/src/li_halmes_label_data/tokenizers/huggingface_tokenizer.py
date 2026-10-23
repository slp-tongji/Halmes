from pathlib import Path

from li_halmes_configurations import HalmesTargetLlm
from tokenizers import Tokenizer

from li_halmes_label_data.supported_tokenizer import SupportedTokenizer


class HuggingfaceTokenizer(SupportedTokenizer):
    def __init__(self, target: HalmesTargetLlm):
        super().__init__(target["id"])
        self.__tokenizer: Tokenizer = Tokenizer.from_file(str(target["model_path"] / "tokenizer.json"))

    def _decode(self, ids: list[int]) -> str:
        return self.__tokenizer.decode(ids)
    
    @property
    def inner_tokenizer(self):
        return self.__tokenizer