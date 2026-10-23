from pathlib import Path
from typing import Literal, TypedDict


class HalmesTargetLlm(TypedDict):
    id: str
    model_type: Literal["qwen3_vl", "intern_vl_3d5"]
    model_path: Path

    do_sample: bool
    temperature: float
    top_p: float
    top_k: float
