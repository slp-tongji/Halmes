import contextlib
import json
from pathlib import Path
import sqlite3
import time
from types import TracebackType
from typing import Any, Literal
import uuid

import csdir
import torch


class HalmesDataManager[T](contextlib.AbstractContextManager):
    def __init__(self, path: Path, throw_if_not_exists: bool) -> None:
        self.__path = path.absolute()

        if throw_if_not_exists and not self.__path.exists():
            raise FileNotFoundError(f"The directory does not exist: {self.__path}")
        
        self.__files_path = csdir.create_directory(self.__path / "files")
        self.__cpu = torch.device("cpu")

        self.__connection = sqlite3.connect(self.__path / "contents.db", check_same_thread=False)

        with self.__connection:
            self.__connection.execute("""
                CREATE TABLE IF NOT EXISTS entries (
                    id TEXT PRIMARY KEY,
                    file TEXT
                )
            """)

    @property
    def path(self):
        return self.__path

    def __enter__(self):
        return self
    
    def __exit__(self, 
                 exc_type: type[BaseException] | None,
                 exc_value: BaseException | None,
                 traceback: TracebackType | None):
        self.__connection.close()
    
    def __get_path(self, file_name: str) -> Path:
        return csdir.create_directory(self.__files_path / file_name[:3]) / f"{file_name}.pt"

    def __save_item(self, item: T):
        while True:
            file_name: str = uuid.uuid4().hex
            file_path: Path = self.__get_path(file_name)
            if file_path.exists():
                continue
            torch.save(item, file_path)
            return file_name

    def __load_item(self, file_name: str) -> T:
        return torch.load(self.__get_path(file_name), 
                          map_location=self.__cpu, 
                          weights_only=True)

    def __unlink_silent(self, path: Path):
        try:
            path.unlink(missing_ok=True)
        except:
            pass

    def insert(self, id: str, item: T):
        file_name: str = self.__save_item(item)
        try:
            with self.__connection:
                _ = self.__connection.execute(
                    "INSERT INTO entries (id, file) VALUES (?, ?)", 
                    (id, file_name,)
                )
        except:
            self.__unlink_silent(self.__get_path(file_name))
            raise
    
    def __select_file_name(self, id: str):
        cursor: sqlite3.Cursor = self.__connection.execute(
            "SELECT file FROM entries WHERE id = ?", (id,)
        )
        row: tuple[str] | None = cursor.fetchone()
        if row is None:
            return None
        return row[0]

    def get(self, id: str) -> tuple[Literal[True], T] | tuple[Literal[False], None]:
        file_name: str | None = self.__select_file_name(id)
        if file_name is None:
            return (False, None)
        return (True, self.__load_item(file_name))

    def update(self, id: str, item: T) -> bool:
        original_file: str | None = self.__select_file_name(id)
        if original_file is None:
            return False

        new_file: str = self.__save_item(item)

        try:
            with self.__connection:
                self.__connection.execute(
                    "UPDATE entries SET file = ? WHERE id = ?",
                    (new_file, id)
                )
        except:
            self.__unlink_silent(self.__get_path(new_file))
            raise

        self.__unlink_silent(self.__get_path(original_file))
        return True

    def upsert(self, id: str, item: T):
        if not self.update(id, item):
            self.insert(id, item)

    def delete(self, id: str) -> bool:
        with self.__connection:
            cursor: sqlite3.Cursor = self.__connection.execute(
                "DELETE FROM entries WHERE id = ? RETURNING file", 
                (id,)
            )
            row: tuple[str] | None = cursor.fetchone()
            
        if row is None:
            return False

        self.__unlink_silent(self.__get_path(row[0]))
        return True
    
    def find(self) -> list[str]:
        query: str = "SELECT id FROM entries"
        cursor: sqlite3.Cursor = self.__connection.execute(query)
        rows: list[tuple[str]] = cursor.fetchall()
        return [r[0] for r in rows]
    
    def count(self) -> int:
        cursor: sqlite3.Cursor = self.__connection.execute("SELECT COUNT(*) FROM entries")
        return cursor.fetchone()[0]

    def cleanup(self, threshold_seconds: int = 300):
        time_threshold: float = time.time() - threshold_seconds
        
        cursor: sqlite3.Cursor = self.__connection.execute("SELECT file FROM entries")
        valid_files: set[str] = {row[0] for row in cursor if row[0]}

        file: Path
        for file in self.__files_path.rglob("*"):
            try:
                if not file.is_file():
                    continue
                if file.stem in valid_files:
                    continue
                if time_threshold < file.stat().st_mtime:
                    continue
                self.__unlink_silent(file)
            except:
                continue

        for file in self.__files_path.rglob("*"):
            try:
                if file.is_dir():
                    file.rmdir()
            except:
                continue
