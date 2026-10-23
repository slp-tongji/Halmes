import json
from pathlib import Path
from typing import Iterable, Sequence, TypedDict

import csfile
from li_data_loader import DatasetInputProvider
from li_image_operating import InMemoryImage
from li_llm_messages import LlmMessage
from li_llm_messages import LlmUserMessage



class BingoInputsSub(DatasetInputProvider):
    class _JsonItem(TypedDict):
        image_path: str
        question: str
        answer: str

    def __init__(self, json_file: Path, images: Path) -> None:
        super().__init__()

        lines: list[BingoInputsSub._JsonItem] = json.loads(csfile.read_all_text(json_file))
        # they seems to repeat the data twice by mistake
        if len(lines) % 2 == 0:
            half_size: int = len(lines) // 2
            half_piece: list[BingoInputsSub._JsonItem] = lines[:half_size]
            if half_piece == lines[half_size:]:
                lines = half_piece
        
        self.__lines = {
            f"bingo_{Path(line["image_path"]).stem}_{line["question"].split(maxsplit=1)[0].lower()}": line
            for line in lines
        }
        self.__images = images

    @property
    def ids(self) -> Iterable[str]:
        return self.__lines.keys()

    def get_image_question_and_answer(self, id: str):
        line: BingoInputsSub._JsonItem = self.__lines[id]
        return (InMemoryImage.from_file(self.__images / line["image_path"]), line["question"], line["answer"])
    
    def get(self, id: str) -> Sequence[LlmMessage]:
        image: InMemoryImage
        question: str
        image, question, _ = self.get_image_question_and_answer(id)
        return [LlmUserMessage((image, question))]
    