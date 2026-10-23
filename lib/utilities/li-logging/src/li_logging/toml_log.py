import contextlib
from pathlib import Path
from types import TracebackType
from typing import Any, Iterable, Literal

import csdir
import csfile
from mime_bytes import MimeBytes
import tomlkit
import tomlkit.container
import tomlkit.items
import tomlkit.toml_document

from li_logging.toml_log_like import TomlLogLike


class TomlLog(contextlib.AbstractContextManager, TomlLogLike):
    def __init__(self, path: Path, path_relative_to: Path) -> None:
        super().__init__()
        
        self.__path = csdir.create_directory(path)
        self.__path_relative_to = path_relative_to

        self.__document = tomlkit.document()
        self.__locked: bool = False
        self.__create_extra_file_in_counter: dict[str, int] = {}

    def __enter__(self):
        return self

    def __exit__(self, 
                 exc_type: type[BaseException] | None,
                 exc_value: BaseException | None,
                 traceback: TracebackType | None):
        self.save_document()
    
    @property
    def name(self):
        return str(self.path.relative_to(self.__path_relative_to))
    
    @property
    def path(self):
        return self.__path
    
    @property
    def path_document(self):
        return self.__path / f"log.toml"
    
    @property
    def document(self):
        return self.__document
    
    def lock(self):
        self.__locked = True
    
    def __check_lock(self):
        if self.__locked:
            raise RuntimeError("The log has been locked.")

    def set_document_value(self, value: Any, key: str | Iterable[tomlkit.items.Key | str]):
        self.__check_lock()

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
        self.__check_lock()
        csfile.write_all_text(self.path_document, self.document.as_string())
        if lock:
            self.lock()

    def __return_path(self, 
                      path_resolved: Path, 
                      return_mode: Literal["absolute", "relative_to_log_file"]):
        match return_mode:
            case "absolute":
                return path_resolved
            case "relative_to_log_file":
                try:
                    return path_resolved.relative_to(self.path, walk_up=True)
                except ValueError:
                    return path_resolved
            case _:
                raise ValueError(f"Unknown return mode '{return_mode}'.")

    def create_sub_log(self, name: str):
        return TomlLog(self.path / name, self.__path_relative_to)

    def create_extra_file(self, file: MimeBytes, 
                          name: str, guess_extension: bool = True,
                          link_if_possible: bool = True,
                          return_mode: Literal["absolute", 
                                               "relative_to_log_file"] = "relative_to_log_file"):
        self.__check_lock()
        
        path: Path = file.to_file(self.__path / name, guess_extension, 
                                  link_mode="prefer_symlink" if link_if_possible else "none")
        return self.__return_path(path, return_mode)
    
    def create_extra_file_in(self, directory: str, file: MimeBytes,
                             link_if_possible: bool = True,
                             return_mode: Literal["absolute", 
                                                  "relative_to_log_file"] = "relative_to_log_file"):
        self.__check_lock()

        file_count: int = self.__create_extra_file_in_counter.get(directory, 0)
        self.__create_extra_file_in_counter[directory] = file_count + 1
        
        path: Path = file.to_file(self.__path / directory / f"{directory[0]}{file_count}", 
                                  link_mode="prefer_symlink" if link_if_possible else "none")
        return self.__return_path(path, return_mode)
    