from dataclasses import dataclass
from typing import AbstractSet, Sequence

from li_data_loader import SimpleDatasetJudgement


@dataclass(frozen=True)
class CocoJudgement(SimpleDatasetJudgement):
    response: str
    words: Sequence[tuple[str, int, bool]]
    
    chair_score: float
    dataset_true_objects: AbstractSet[str]