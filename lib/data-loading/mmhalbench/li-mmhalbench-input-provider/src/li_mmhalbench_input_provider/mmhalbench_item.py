from pathlib import Path
from typing import NamedTuple, Sequence

from li_image_operating import InMemoryImage

from li_mmhalbench_input_provider.mmhalbench_response_template_item import MmhalbenchResponseTemplateItem


class MmhalbenchItem(NamedTuple):
    question: str
    image_path: Path
    answer: str
    image_content: tuple[str, ...]

    def get_image(self):
        return InMemoryImage.from_file(self.image_path)
    
    @staticmethod
    def from_response_template(item: MmhalbenchResponseTemplateItem, images: Path):
        return MmhalbenchItem(item["question"], 
                              images / item["image_src"].rsplit("/", 1)[-1], 
                              item["gt_answer"],
                              tuple(item["image_content"]))
    