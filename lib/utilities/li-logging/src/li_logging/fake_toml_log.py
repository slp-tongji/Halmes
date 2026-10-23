import contextlib
from types import TracebackType
from typing import Any, Iterable, Literal

import tomlkit.container
import tomlkit.toml_document

from li_logging.toml_log_like import TomlLogLike
from mime_bytes import MimeBytes
import tomlkit
import tomlkit.items


class FakeTomlLog(contextlib.AbstractContextManager, TomlLogLike):
    def __init__(self) -> None:
        super().__init__()
        self.__document = tomlkit.document()

    def __enter__(self):
        return self

    def __exit__(self, 
                 exc_type: type[BaseException] | None,
                 exc_value: BaseException | None,
                 traceback: TracebackType | None):
        pass
    
    @property
    def name(self):
        return None
    
    @property
    def path(self):
        return None
    
    @property
    def path_document(self):
        return None
    
    @property
    def document(self):
        return self.__document
    
    def lock(self):
        pass
    
    def set_document_value(self, value: Any, key: str | Iterable[tomlkit.items.Key | str]):
        if isinstance(key, str):
            keys: list[tomlkit.items.Key | str] = [key]
        else:
            keys = list(key)
        last_key: tomlkit.items.Key | str = keys.pop()
        
        final_append: tuple[tomlkit.toml_document.TOMLDocument | tomlkit.items.Table, 
                            tomlkit.items.Table, 
                            tomlkit.items.Key | str] | None = None

        current: tomlkit.toml_document.TOMLDocument | tomlkit.items.Table = self.__document
        k: tomlkit.items.Key | str
        for k in keys:
            if k not in current:
                new_table: tomlkit.items.Table = tomlkit.table(True)
                if final_append is None:
                    final_append = (current, new_table, k)
                else:
                    current.append(k, new_table)
                current = new_table
            else:
                next: tomlkit.items.Item | tomlkit.container.Container = current[k]
                if isinstance(next, tomlkit.items.Table):
                    current = next
                else:
                    raise ValueError("Part of the given key has been occupied by a non-table item.")
                
        if last_key in current:
            raise ValueError("The given key has been occupied.")

        current.append(last_key, tomlkit.item(value))

        if final_append is not None:
            final_append[0].append(final_append[2], final_append[1])

    def save_document(self, lock: bool = True):
        pass

    def create_sub_log(self, name: str):
        return FakeTomlLog()

    def create_extra_file(self, file: MimeBytes, 
                          name: str, guess_extension: bool = True,
                          link_if_possible: bool = True,
                          return_mode: Literal["absolute", 
                                               "relative_to_log_file"] = "relative_to_log_file"):
        return None
    
    def create_extra_file_in(self, directory: str, file: MimeBytes,
                             link_if_possible: bool = True,
                             return_mode: Literal["absolute", 
                                                  "relative_to_log_file"] = "relative_to_log_file"):
        return None
    