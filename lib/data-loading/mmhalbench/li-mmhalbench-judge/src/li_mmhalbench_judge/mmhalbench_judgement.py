import dataclasses
from typing import Literal

from li_data_loader import SimpleDatasetJudgement


@dataclasses.dataclass(frozen=True)
class MmhalbenchJudgement(SimpleDatasetJudgement):
    score: Literal[0, 1, 2, 3, 4, 5, 6]
