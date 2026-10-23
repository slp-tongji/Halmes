import json
from pathlib import Path
from typing import Iterable, Sequence, TypedDict

import csfile
from li_data_loader import DatasetInputProvider
from li_image_operating import InMemoryImage
from li_llm_messages import LlmMessage
from li_llm_messages import LlmUserMessage

from li_mmhalbench_input_provider.mmhalbench_item import MmhalbenchItem
from li_mmhalbench_input_provider.mmhalbench_response_template_item import MmhalbenchResponseTemplateItem


class MmhalbenchInputs(DatasetInputProvider):
    def __init__(self, path: Path) -> None:
        super().__init__()

        images: Path = path / "images"
        response_template: list[MmhalbenchResponseTemplateItem]
        response_template = json.loads(csfile.read_all_text(path / "response_template.json"))
        self.__data: dict[str, MmhalbenchItem] = {
            f"mmhalbench_{r["image_id"]}": MmhalbenchItem.from_response_template(r, images)
            for r in response_template
        }

    @property
    def ids(self) -> Iterable[str]:
        return self.__data.keys()

    def get_item(self, id: str) -> MmhalbenchItem:
        return self.__data[id]

    def get(self, id: str) -> Sequence[LlmMessage]:
        item: MmhalbenchItem = self.get_item(id)
        return [LlmUserMessage((item.get_image(), item.question))]
    