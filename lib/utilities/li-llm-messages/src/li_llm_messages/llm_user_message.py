from typing import Callable, NamedTuple

from li_image_operating import InMemoryImage


class LlmUserMessage(NamedTuple):
    content: tuple[str | InMemoryImage, ...]

    def modify_images(self, modification: Callable[[InMemoryImage], InMemoryImage]):
        result: list[str | InMemoryImage] = []
        block: str | InMemoryImage
        for block in self.content:
            if isinstance(block, InMemoryImage):
                result.append(modification(block))
            else:
                result.append(block)
        return LlmUserMessage(tuple(result))
    