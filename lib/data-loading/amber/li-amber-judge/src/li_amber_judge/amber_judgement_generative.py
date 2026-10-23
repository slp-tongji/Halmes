from dataclasses import dataclass
from typing import AbstractSet, Sequence

from li_data_loader import SimpleDatasetJudgement


@dataclass(frozen=True)
class AmberJudgementGenerative(SimpleDatasetJudgement):
    found_truth: AbstractSet[str]
    annotation_truth: Sequence[str]

    found_hallu: AbstractSet[str]
    annotation_hallu: Sequence[str]
    
    non_truth_objects: Sequence[str]
    all_considered_objects: Sequence[str]

    truth_cover_score: float
    hallu_cover_score: float
    chair_score: float
