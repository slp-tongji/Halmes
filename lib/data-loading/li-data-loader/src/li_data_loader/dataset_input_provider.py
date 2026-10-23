import abc
import itertools
from random import Random
from typing import Callable, Iterable, Sequence

from li_data_loader.dataset_input_with_id import DatasetInputWithId
from li_llm_messages import LlmMessage


class DatasetInputProvider(abc.ABC):
    @property
    @abc.abstractmethod
    def ids(self) -> Iterable[str]:
        ...

    @abc.abstractmethod
    def get(self, id: str) -> Sequence[LlmMessage]:
        ...
    
    def enumerate(self) -> Iterable[DatasetInputWithId]:
        id: str
        for id in self.ids:
            yield DatasetInputWithId(id, self.get(id))
    
    def merge(self, other: 'DatasetInputProvider') -> 'DatasetInputProvider':
        return _MergedDatasetInputProvider([self, other])
    
    def merge_many(self, others: Iterable['DatasetInputProvider']) -> 'DatasetInputProvider':
        return _MergedDatasetInputProvider([self] + list(others))
    
    def slice(self, start: int, stop: int | None) -> 'DatasetInputProvider':
        return _SlicedDatasetInputProvider(self, start, stop)

    def where(self, predicate: Callable[[str], bool]) -> 'DatasetInputProvider':
        return _FilteredDatasetInputProvider(self, predicate)
    
    @staticmethod
    def empty() -> 'DatasetInputProvider':
        return _EmptyDatasetInputProvider()


class _MergedDatasetInputProvider(DatasetInputProvider):
    def __init__(self, providers: Iterable[DatasetInputProvider]) -> None:
        super().__init__()

        self.__ids: dict[str, DatasetInputProvider] = {}
        provider: DatasetInputProvider
        for provider in providers:
            if isinstance(provider, _MergedDatasetInputProvider):
                self.__ids.update(provider.__ids)
            else:
                self.__ids.update({id: provider for id in provider.ids})

    @property
    def ids(self) -> Iterable[str]:
        return self.__ids.keys()

    def get(self, id: str) -> Sequence[LlmMessage]:
        return self.__ids[id].get(id)

class _EmptyDatasetInputProvider(DatasetInputProvider):
    @property
    def ids(self) -> Iterable[str]:
        return []

    def get(self, id: str) -> Sequence[LlmMessage]:
        raise ValueError(f"Cannot find '{id}'.")

class _SlicedDatasetInputProvider(DatasetInputProvider):
    def __init__(self, provider: DatasetInputProvider, start: int, stop: int | None) -> None:
        super().__init__()
        self.__provider = provider
        self.__ids = list(itertools.islice(provider.ids, start, stop))
    
    @property
    def ids(self) -> Iterable[str]:
        return self.__ids

    def get(self, id: str) -> Sequence[LlmMessage]:
        return self.__provider.get(id)

class _FilteredDatasetInputProvider(DatasetInputProvider):
    def __init__(self, provider: DatasetInputProvider, filter_: Callable[[str], bool]) -> None:
        super().__init__()
        self.__provider = provider
        self.__ids = list(filter(filter_, provider.ids))

    @property
    def ids(self) -> Iterable[str]:
        return self.__ids

    def get(self, id: str) -> Sequence[LlmMessage]:
        return self.__provider.get(id)
