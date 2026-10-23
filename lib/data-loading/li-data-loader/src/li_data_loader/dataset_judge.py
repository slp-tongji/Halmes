import abc
from typing import Generic, Iterable, Never, NoReturn, TypeVar

from li_llm_messages import LlmAssistantMessage
from li_logging import TomlLogLike

from li_data_loader.dataset_judgement_protocol import DatasetJudgementProtocol


_T = TypeVar("_T", bound=DatasetJudgementProtocol, covariant=True)    # noqa: WVarDec01

class DatasetJudge(abc.ABC, Generic[_T]):
    @property
    @abc.abstractmethod
    def ids(self) -> Iterable[str]:
        ...

    @abc.abstractmethod
    def judge(self, id: str, response: LlmAssistantMessage, log: TomlLogLike) -> _T:
        ...

    def merge[TOther: DatasetJudgementProtocol](self, other: 'DatasetJudge[TOther]') -> 'DatasetJudge[_T | TOther]':
        return _MergedDatasetJudge([self, other])
    
    def merge_many[TOther: DatasetJudgementProtocol](self, others: Iterable['DatasetJudge[TOther]']) -> 'DatasetJudge[_T | TOther]':
        return _MergedDatasetJudge([self] + list(others))
    
    @staticmethod
    def empty() -> 'DatasetJudge[Never]':
        return _EmptyDatasetJudge()


class _MergedDatasetJudge[T: DatasetJudgementProtocol](DatasetJudge[T]):
    def __init__(self, providers: Iterable[DatasetJudge[T]]) -> None:
        super().__init__()

        self.__ids: dict[str, DatasetJudge] = {}
        provider: DatasetJudge
        for provider in providers:
            if isinstance(provider, _MergedDatasetJudge):
                self.__ids.update(provider.__ids)
            else:
                self.__ids.update({id: provider for id in provider.ids})

    @property
    def ids(self) -> Iterable[str]:
        return self.__ids.keys()

    def judge(self, id: str, response: LlmAssistantMessage, log: TomlLogLike) -> T:
        return self.__ids[id].judge(id, response, log)

class _EmptyDatasetJudge(DatasetJudge[Never]):
    @property
    def ids(self) -> Iterable[str]:
        return []

    def judge(self, id: str, response: LlmAssistantMessage, log: TomlLogLike) -> NoReturn:
        raise ValueError(f"Cannot deal with '{id}'.")
