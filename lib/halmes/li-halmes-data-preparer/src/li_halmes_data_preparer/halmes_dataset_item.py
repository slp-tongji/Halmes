from typing import TypedDict

from statictorch import Tensor2d


class HalmesDatasetItem(TypedDict):
    feature: Tensor2d
    is_hallucination: bool
