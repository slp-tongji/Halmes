import json
from pathlib import Path
from typing import Iterable, Sequence, TypedDict

import csfile
from li_data_loader import DatasetInputProvider
from li_image_operating import InMemoryImage
from li_llm_messages import LlmMessage
from li_llm_messages import LlmUserMessage


class CocoInputs(DatasetInputProvider):
    class _ImagesItem(TypedDict):
        license: int
        file_name: str
        coco_url: str
        height: int
        width: int
        date_captured: str
        flicker_url: str
        id: int

    def __init__(self, path_captions_json: Path, path_images: Path) -> None:
        super().__init__()

        images: Sequence[CocoInputs._ImagesItem] = json.loads(csfile.read_all_text(path_captions_json))["images"]
        self.__images: dict[str, Path] = {
            f"coco_{i["id"]}": path_images / i["file_name"]
            for i in images
        }
        
    @property
    def ids(self) -> Iterable[str]:
        return self.__images.keys()

    def get(self, id: str) -> Sequence[LlmMessage]:
        return [
            LlmUserMessage((
                InMemoryImage.from_file(self.__images[id]),
                # COCO does not say how to use it for large language models, but of course we need a text prompt.
                "Please describe this image in detail."
            ))
        ]