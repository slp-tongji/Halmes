from pathlib import Path

from li_basic_utilities import StaticClass
from li_openai import OpenaiLlmParameter, RemoteOpenaiLlm


class HalmesToolConfiguration(StaticClass):
    @staticmethod
    def __tools() -> Path:
        # This is just for convenience; feel free to edit the functions below directly.
        return Path("../../managing/links/tools").absolute()

    @staticmethod
    def nltk() -> Path:
        return HalmesToolConfiguration.__tools() / "nltk/nltk_data/packages"

    @staticmethod
    def spacy_encoreweblg() -> Path:
        return HalmesToolConfiguration.__tools() / "spacy/en_core_web_lg/en_core_web_lg-3.8.0"
    