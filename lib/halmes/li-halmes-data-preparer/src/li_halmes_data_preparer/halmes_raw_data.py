from typing import Sequence, TypedDict
from li_halmes_model import HalmesFeature
from statictorch import Tensor1d


class HalmesRawData(TypedDict):
    data_id: str
    output: Tensor1d

    features: Sequence[HalmesFeature]
    