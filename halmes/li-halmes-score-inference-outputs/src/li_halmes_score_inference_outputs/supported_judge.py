import abc
from typing import AbstractSet, Iterable, Mapping, NamedTuple

from li_llm_messages import LlmAssistantMessage
from li_logging import TomlLogLike


class SupportedJudge(abc.ABC):
    @property
    @abc.abstractmethod
    def ids(self) -> Iterable[str]:
        ...

    @abc.abstractmethod
    def get_database_id(self, id: str) -> str:
        ...

    @abc.abstractmethod
    def judge(self, id: str, response: LlmAssistantMessage, log: TomlLogLike) -> Mapping[str, str | float | int]:
        ...
    
    def merge(self, other: 'SupportedJudge') -> 'SupportedJudge':
        return _MergedDatasetJudge([self, other])
    
    def merge_many(self, others: Iterable['SupportedJudge']) -> 'SupportedJudge':
        return _MergedDatasetJudge([self] + list(others))
    
    @staticmethod
    def empty() -> 'SupportedJudge':
        return _EmptyDatasetJudge()


class _MergedDatasetJudge(SupportedJudge):
    def __init__(self, providers: Iterable[SupportedJudge]) -> None:
        super().__init__()

        self.__ids: dict[str, SupportedJudge] = {}
        provider: SupportedJudge
        for provider in providers:
            if isinstance(provider, _MergedDatasetJudge):
                self.__ids.update(provider.__ids)
            else:
                self.__ids.update({id: provider for id in provider.ids})

    @property
    def ids(self) -> Iterable[str]:
        return self.__ids.keys()

    def get_database_id(self, id: str) -> str:
        return self.__ids[id].get_database_id(id)

    def judge(self, id: str, response: LlmAssistantMessage, log: TomlLogLike) -> Mapping[str, str | float | int]:
        return self.__ids[id].judge(id, response, log)


class _EmptyDatasetJudge(SupportedJudge):
    @property
    def ids(self) -> Iterable[str]:
        return []

    def get_database_id(self, id: str) -> str:
        raise ValueError(f"Cannot deal with '{id}'.")

    def judge(self, id: str, response: LlmAssistantMessage, log: TomlLogLike) -> Mapping[str, str | float | int]:
        raise ValueError(f"Cannot deal with '{id}'.")
