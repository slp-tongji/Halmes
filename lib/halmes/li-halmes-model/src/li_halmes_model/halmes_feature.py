from typing import TypedDict
from statictorch import Tensor0d, Tensor2d


class HalmesFeature(TypedDict):
    var: Tensor2d
    svar: Tensor0d
    attention_text: Tensor0d
    attention_generated_text: Tensor0d
    text_visual_ratio: Tensor0d
    outgoing_entropy: Tensor0d
    