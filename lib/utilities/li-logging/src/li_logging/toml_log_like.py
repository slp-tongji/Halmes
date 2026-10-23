from pathlib import Path
from types import TracebackType
from typing import Any, Iterable, Literal, Protocol, Self

from mime_bytes import MimeBytes
import tomlkit
import tomlkit.items


class TomlLogLike(Protocol):
    def __enter__(self) -> Self:
        ...

    def __exit__(self, 
                 exc_type: type[BaseException] | None,
                 exc_value: BaseException | None,
                 traceback: TracebackType | None) -> None:
        ...
    
    @property
    def name(self) -> str | None:
        ...
    
    @property
    def path(self) -> Path | None:
        ...
    
    @property
    def path_document(self) -> Path | None:
        ...
    
    @property
    def document(self) -> tomlkit.TOMLDocument:
        ...
    
    def lock(self) -> None:
        ...
    
    def set_document_value(self, value: Any, key: str | Iterable[tomlkit.items.Key | str]) -> None:
        ...

    def save_document(self, lock: bool = True) -> None:
        ...

    def create_sub_log(self, name: str) -> Self:
        ...

    def create_extra_file(self, file: MimeBytes, 
                          name: str, guess_extension: bool = True,
                          link_if_possible: bool = True,
                          return_mode: Literal["absolute", 
                                               "relative_to_log_file"] = "relative_to_log_file") -> Path | None:
        ...
    
    def create_extra_file_in(self, directory: str, file: MimeBytes,
                             link_if_possible: bool = True,
                             return_mode: Literal["absolute", 
                                                  "relative_to_log_file"] = "relative_to_log_file") -> Path | None:
        ...
    