from pathlib import Path
from random import Random
import csdir
from li_basic_utilities import StringRandom

from li_logging.toml_log import TomlLog


class TomlLogger:
    def __init__(self, 
                 directory: Path | str, 
                 random_seed: int | str | None = None,
                 raise_if_not_empty: bool = True) -> None:
        directory = Path(directory).resolve()
        if directory.exists() and raise_if_not_empty:
            if any(directory.iterdir()):
                raise RuntimeError(f"The directory '{directory}' is not empty.")
        
        self.__directory = csdir.create_directory(directory)
        self.__random = StringRandom(Random(random_seed), 12, False)

    @property
    def directory(self):
        return self.__directory

    def new_log(self) -> TomlLog:
        id: str = self.__random.next()
        path: Path = self.__directory / id[:len(id) // 4]
        return TomlLog(path / id, path)
