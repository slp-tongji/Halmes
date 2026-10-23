from contextlib import AbstractContextManager
import contextlib
import csv
import multiprocessing
from pathlib import Path
import threading
from typing import Any, Iterable, Literal
import typing

import csdir

if typing.TYPE_CHECKING:
    import _typeshed

class CsvFileWriter(AbstractContextManager):
    def __init__(self, 
                 path: Path, 
                 mode: '_typeshed.OpenTextModeWriting | _typeshed.OpenTextModeUpdating' = "w",
                 create_directory: bool = True,
                 lock_type: Literal["none", "threading", "multiprocessing"] = "threading") -> None:
        self.__path = path.resolve()
        if create_directory:
            csdir.create_directory(self.__path.parent)
        self.__file = open(self.__path, mode=mode, newline='', encoding='utf8')
        self.__writer = csv.writer(self.__file)

        match lock_type:
            case "threading":
                self.__lock = threading.Lock()
            case "multiprocessing":
                self.__lock = multiprocessing.Lock()
            case "none":
                self.__lock = contextlib.nullcontext()
            case _:
                raise RuntimeError(f"Unsupported lock type '{lock_type}'.")

    def path(self):
        return self.__path

    def write_row(self, row: Iterable[str | float | bool]):
        with self.__lock:
            self.__writer.writerow(row)
            self.__file.flush()

    def write_rows(self, rows: Iterable[Iterable[str | float | bool]]):
        with self.__lock:
            self.__writer.writerows(rows)
            self.__file.flush()

    def close(self):
        with self.__lock:
            self.__file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any):
        self.close()
    