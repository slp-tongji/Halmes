from concurrent.futures import ProcessPoolExecutor
import contextlib
from pathlib import Path
from types import TracebackType

import nltk


class CrossProcessNltkWrapper(contextlib.AbstractContextManager):
    def __init__(self, path: Path):
        self.__process = ProcessPoolExecutor(
            max_workers=1, 
            initializer=CrossProcessNltkWrapper._initialize,
            initargs=(str(path),))
    
    def __enter__(self):
        return self

    def __exit__(self, 
                 exc_type: type[BaseException] | None,
                 exc_value: BaseException | None,
                 traceback: TracebackType | None):
        self.__process.shutdown()

    @staticmethod
    def _initialize(path: str):
        nltk.data.path = [path]

    @staticmethod
    def _pos_tag(tokens: list[str], tagset: str | None, lang: str) -> list[tuple[str, str]]:
        return nltk.pos_tag(tokens, tagset, lang)

    def pos_tag(self, tokens: list[str], tagset: str | None = None, lang: str = "eng"):
        return self.__process.submit(CrossProcessNltkWrapper._pos_tag, tokens, tagset, lang).result()
    
    @staticmethod
    def _word_tokenize(text: str, language: str, preserve_line: bool) -> list[str]:
        return nltk.word_tokenize(text, language, preserve_line)

    def word_tokenize(self, text: str, language: str = "english", preserve_line: bool = False) -> list[str]:
        return self.__process.submit(CrossProcessNltkWrapper._word_tokenize, text, language, preserve_line).result()
    
    @staticmethod
    def _word_net_lemmatizer_lemmatize(word: str, pos: str) -> str:
        return nltk.stem.WordNetLemmatizer().lemmatize(word, pos)
    
    def word_net_lemmatizer_lemmatize(self, word: str, pos: str = "n") -> str:
        return self.__process.submit(CrossProcessNltkWrapper._word_net_lemmatizer_lemmatize, word, pos).result()
    
    @staticmethod
    def _word_net_lemmatizer_lemmatize_many(word_and_pos_list: list[tuple[str, str]]) -> list[str]:
        lemmatizer: nltk.stem.WordNetLemmatizer = nltk.stem.WordNetLemmatizer()
        result: list[str] = []
        for word, pos in word_and_pos_list:
            result.append(lemmatizer.lemmatize(word, pos))
        return result
    
    def word_net_lemmatizer_lemmatize_many(self, word_and_pos_list: list[tuple[str, str]]) -> list[str]:
        return self.__process.submit(CrossProcessNltkWrapper._word_net_lemmatizer_lemmatize_many, 
                                     word_and_pos_list).result()
    