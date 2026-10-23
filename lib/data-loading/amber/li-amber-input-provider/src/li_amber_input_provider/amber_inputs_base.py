import json
from pathlib import Path
from typing import Iterable

import csfile
from li_data_loader import DatasetInputProvider
from li_image_operating import InMemoryImage

from li_amber_input_provider.amber_query import AmberQuery


class AmberInputsBase(DatasetInputProvider):
    def __init__(self, path_image: Path, path_query: Path) -> None:
        super().__init__()
        self.__path_image: Path = path_image
        queries: list[AmberQuery] = json.loads(csfile.read_all_text(path_query))
        self.__queries = {f"amber_{q["id"]}": q for q in queries}

    @property
    def ids(self) -> Iterable[str]:
        return self.__queries.keys()

    def get_image_and_query(self, id: str):
        query: AmberQuery = self.__queries[id]
        return (InMemoryImage.from_file(self.__path_image / query["image"]), query["query"])
    
    def get_query(self, id: str):
        query: AmberQuery = self.__queries[id]
        return query["query"]
    